import asyncio
import time
from dataclasses import dataclass

from fastapi import WebSocket

from app.core.config import settings
from app.core.database import SessionLocal
from app.services.live_quote_service import (
    LiveQuoteService,
)


@dataclass(
    frozen=True,
    slots=True,
)
class LiveQuoteSubscriber:
    websocket: WebSocket
    symbols: tuple[str, ...]


class LiveQuoteHub:
    def __init__(self):
        self._subscribers: dict[
            int,
            LiveQuoteSubscriber,
        ] = {}

        self._lock = asyncio.Lock()

        self._polling_task: (
            asyncio.Task | None
        ) = None

    async def subscribe(
        self,
        websocket: WebSocket,
        symbols: list[str],
    ) -> int:
        subscriber_id = id(
            websocket
        )

        subscriber = (
            LiveQuoteSubscriber(
                websocket=websocket,
                symbols=tuple(symbols),
            )
        )

        async with self._lock:
            self._subscribers[
                subscriber_id
            ] = subscriber

            if (
                self._polling_task
                is None
                or self._polling_task.done()
            ):
                self._polling_task = (
                    asyncio.create_task(
                        self._poll_loop()
                    )
                )

        print(
            "[LIVE][WS-SUBSCRIBE] "
            f"id={subscriber_id} "
            f"symbols={len(symbols)} "
            f"clients={await self.client_count()}",
            flush=True,
        )

        return subscriber_id

    async def unsubscribe(
        self,
        subscriber_id: int,
    ) -> None:
        task_to_cancel: (
            asyncio.Task | None
        ) = None

        async with self._lock:
            self._subscribers.pop(
                subscriber_id,
                None,
            )

            if (
                not self._subscribers
                and self._polling_task
                is not None
            ):
                task_to_cancel = (
                    self._polling_task
                )

                self._polling_task = None

        if (
            task_to_cancel is not None
            and task_to_cancel
            is not asyncio.current_task()
        ):
            task_to_cancel.cancel()

        print(
            "[LIVE][WS-UNSUBSCRIBE] "
            f"id={subscriber_id} "
            f"clients={await self.client_count()}",
            flush=True,
        )

    async def client_count(
        self,
    ) -> int:
        async with self._lock:
            return len(
                self._subscribers
            )

    async def _snapshot(
        self,
    ) -> dict[
        int,
        LiveQuoteSubscriber,
    ]:
        async with self._lock:
            return dict(
                self._subscribers
            )

    @staticmethod
    def _chunks(
        values: list[str],
        size: int,
    ):
        for start in range(
            0,
            len(values),
            size,
        ):
            yield values[
                start:
                start + size
            ]

    async def _poll_quotes(
        self,
        symbols: list[str],
    ) -> dict[str, dict]:
        db = SessionLocal()

        try:
            service = LiveQuoteService(
                db
            )

            result: dict[
                str,
                dict,
            ] = {}

            for chunk in self._chunks(
                symbols,
                200,
            ):
                quotes = (
                    await service
                    .get_quotes(
                        chunk,
                        force=True,
                    )
                )

                for quote in quotes:
                    payload = (
                        quote.model_dump()
                    )

                    result[
                        quote.stockCode
                    ] = payload

            return result

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

    async def _send_to_subscriber(
        self,
        subscriber_id: int,
        subscriber: (
            LiveQuoteSubscriber
        ),
        quotes: dict[str, dict],
    ) -> int | None:
        payload_quotes = [
            quotes[symbol]
            for symbol
            in subscriber.symbols
            if symbol in quotes
        ]

        if not payload_quotes:
            return None

        try:
            await subscriber.websocket.send_json(
                {
                    "type": "quotes",
                    "quotes": payload_quotes,
                }
            )

            return None

        except Exception:
            return subscriber_id

    async def _remove_dead(
        self,
        subscriber_ids: list[int],
    ) -> None:
        if not subscriber_ids:
            return

        async with self._lock:
            for subscriber_id in (
                subscriber_ids
            ):
                self._subscribers.pop(
                    subscriber_id,
                    None,
                )

    async def _broadcast(
        self,
        subscribers: dict[
            int,
            LiveQuoteSubscriber,
        ],
        quotes: dict[str, dict],
    ) -> None:
        results = await asyncio.gather(
            *[
                self._send_to_subscriber(
                    subscriber_id,
                    subscriber,
                    quotes,
                )
                for (
                    subscriber_id,
                    subscriber,
                ) in subscribers.items()
            ],
            return_exceptions=False,
        )

        dead_ids = [
            subscriber_id
            for subscriber_id in results
            if subscriber_id is not None
        ]

        await self._remove_dead(
            dead_ids
        )

    async def _poll_loop(
        self,
    ) -> None:
        print(
            "[LIVE][WS-POLLER] START",
            flush=True,
        )

        try:
            while True:
                started_at = (
                    time.monotonic()
                )

                subscribers = (
                    await self._snapshot()
                )

                if not subscribers:
                    return

                symbols = list(
                    dict.fromkeys(
                        symbol
                        for subscriber
                        in subscribers.values()
                        for symbol
                        in subscriber.symbols
                    )
                )

                try:
                    quotes = (
                        await self._poll_quotes(
                            symbols
                        )
                    )

                    await self._broadcast(
                        subscribers,
                        quotes,
                    )

                    print(
                        "[LIVE][WS-PUSH] "
                        f"clients={len(subscribers)} "
                        f"symbols={len(symbols)} "
                        f"quotes={len(quotes)}",
                        flush=True,
                    )

                except Exception as e:
                    print(
                        "[LIVE][WS-POLL-FAIL] "
                        f"{type(e).__name__}: "
                        f"{e}",
                        flush=True,
                    )

                elapsed = (
                    time.monotonic()
                    - started_at
                )

                sleep_seconds = max(
                    0.0,
                    float(
                        settings
                        .live_quote_poll_seconds
                    )
                    - elapsed,
                )

                await asyncio.sleep(
                    sleep_seconds
                )

        except asyncio.CancelledError:
            pass

        finally:
            print(
                "[LIVE][WS-POLLER] STOP",
                flush=True,
            )


live_quote_hub = LiveQuoteHub()
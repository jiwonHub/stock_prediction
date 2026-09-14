from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import (
    ConfigurationError,
    ExternalApiError,
)
from app.schemas.live_quote import (
    LiveQuoteResponse,
)
from app.services.live_quote_service import (
    LiveQuoteService,
)
from app.services.live_quote_hub import (
    live_quote_hub,
)


router = APIRouter(
    prefix="/live",
    tags=[
        "live",
    ],
)


@router.get(
    "/quotes",
    response_model=list[
        LiveQuoteResponse
    ],
)
async def get_live_quotes(
    symbols: str = Query(
        min_length=1,
        max_length=1600,
    ),
    force: bool = False,
    db: Session = Depends(
        get_db
    ),
):
    stock_codes = [
        value.strip()
        for value in symbols.split(
            ","
        )
        if value.strip()
    ]

    try:
        return await LiveQuoteService(
            db
        ).get_quotes(
            stock_codes,
            force=force,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e

    except ConfigurationError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    except ExternalApiError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e
    
@router.websocket(
    "/quotes/ws",
)
async def stream_live_quotes(
    websocket: WebSocket,
):
    raw_symbols = (
        websocket.query_params.get(
            "symbols",
            "",
        )
    )

    symbols = list(
        dict.fromkeys(
            value.strip()
            for value
            in raw_symbols.split(",")
            if value.strip()
        )
    )

    if not symbols:
        await websocket.close(
            code=1008
        )
        return

    if len(symbols) > 200:
        await websocket.close(
            code=1008
        )
        return

    await websocket.accept()

    subscriber_id = (
        await live_quote_hub
        .subscribe(
            websocket,
            symbols,
        )
    )

    await websocket.send_json(
        {
            "type": "connected",
            "symbols": symbols,
        }
    )

    try:
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        pass

    finally:
        await live_quote_hub.unsubscribe(
            subscriber_id
        )
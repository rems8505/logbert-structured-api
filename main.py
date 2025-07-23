import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
print(sys.path)
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


from fastapi.responses import Response # Prometheus_Monitoring
from prometheus_client import generate_latest # Prometheus_Monitoring

# from app.api import api_router
from config import settings
import bert

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

from bert import router as bert_router #Prometheus_Monitoring
root_router = APIRouter()
app.include_router(bert.router, prefix="/bert", tags=["bert"])

@app.get("/metrics") #Prometheus_Monitoring
async def get_metrics(): #Prometheus_Monitoring
    return Response(media_type="text/plain", content=generate_latest()) #Prometheus_Monitoring



# Initialize BERT model on startup
@app.on_event("startup")
async def startup_event():
    bert.load_bert_model()

@root_router.get("/")
def index(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
         "<div>"
        "Check the API health here: <a href='/bert'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)


# app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 

    ## localhost--> 127.0.0.0
    ## host --> 0.0.0.0 allows all host

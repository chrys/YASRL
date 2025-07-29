import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.yasrl.api:app",  # Pass as string instead of importing
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True  # This will now work properly
    )
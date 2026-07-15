# Teo.AI Mobile

Flutter client for Teo.AI — matches the web UI (dark + yellow barcode theme).

## Setup

```bash
cd app/mobile
flutter pub get
```

## Run

Start the backend first:

```bash
cd app/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then run the app:

```bash
# Android emulator (default API: 10.0.2.2:8000)
flutter run

# Physical device — point to your machine's LAN IP
flutter run --dart-define=TEO_API=http://192.168.1.100:8000

# iOS simulator (localhost works)
flutter run --dart-define=TEO_API=http://127.0.0.1:8000
```

## Notes

- Personal project — not production-ready.
- API URL is set via `TEO_API` dart-define at build time.
- Conversation history is sent to `/api/chat` (Phase 2 contract).

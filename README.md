# Chatbot App

A simple full-stack chatbot built with Node.js and vanilla HTML/CSS/JS.

## Run

1. Open terminal in this folder.
2. Start server:

```bash
npm start
```

3. Open `http://localhost:3000` in your browser.

## API

- `POST /api/chat`
  - Body: `{ "message": "Hello" }`
  - Response: `{ "reply": "..." }`

- `GET /health`
  - Response: `{ "ok": true }`

## Notes

- The bot currently uses simple rule-based responses in `server.js`.
- You can later connect it to OpenAI/Gemini/other LLM APIs by replacing `botReply()`.

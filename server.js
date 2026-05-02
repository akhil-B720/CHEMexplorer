const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = Number(process.env.PORT) || 3000;
const HOST = process.env.HOST || "0.0.0.0";

function sendJson(res, statusCode, payload) {
  const data = JSON.stringify(payload);
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(data)
  });
  res.end(data);
}

function parseBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";

    req.on("data", (chunk) => {
      body += chunk.toString("utf8");
      if (body.length > 1e6) {
        reject(new Error("Payload too large"));
      }
    });

    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (err) {
        reject(new Error("Invalid JSON payload"));
      }
    });

    req.on("error", reject);
  });
}

function botReply(message) {
  const text = (message || "").toLowerCase().trim();

  if (!text) return "Please type a message so I can respond.";
  if (text.includes("hello") || text.includes("hi")) {
    return "Hey! I am your chatbot. Ask me anything.";
  }
  if (text.includes("your name")) {
    return "I am CHEMexplorer Bot, built with Node.js.";
  }
  if (text.includes("time")) {
    return `Current server time is ${new Date().toLocaleString()}.`;
  }
  if (text.includes("help")) {
    return "Try asking about my name, say hello, or ask for the current time.";
  }

  return `You said: "${message}". I am a simple rule-based bot, but you can connect me to an AI API later.`;
}

const server = http.createServer(async (req, res) => {
  const { method, url } = req;

  if (method === "GET" && (url === "/" || url === "/index.html")) {
    const htmlPath = path.join(__dirname, "public", "index.html");
    fs.readFile(htmlPath, (err, data) => {
      if (err) {
        sendJson(res, 500, { error: "Failed to load UI" });
        return;
      }
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(data);
    });
    return;
  }

  if (method === "POST" && url === "/api/chat") {
    try {
      const body = await parseBody(req);
      const message = typeof body.message === "string" ? body.message : "";
      const reply = botReply(message);
      sendJson(res, 200, { reply });
    } catch (err) {
      sendJson(res, 400, { error: err.message });
    }
    return;
  }

  if (method === "GET" && url === "/health") {
    sendJson(res, 200, { ok: true });
    return;
  }

  sendJson(res, 404, { error: "Not found" });
});

server.listen(PORT, HOST, () => {
  console.log(`Chatbot listening on port ${PORT} (host ${HOST})`);
});

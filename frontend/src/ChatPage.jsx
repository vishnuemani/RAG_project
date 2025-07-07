import React, { useState, useRef, useEffect } from 'react';

export default function ChatPage() {
  const [messages, setMessages] = useState([]); // {author, text, timestamp, sources?}
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = async () => {
    const question = input.trim();
    if (!question || loading) return;
    const time = new Date().toISOString();
    setMessages(m => [...m, { author: 'user', text: question, timestamp: time }]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${process.env.REACT_APP_API_URL || ''}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      if (!res.ok || !res.body) throw new Error('Network response was not ok');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let aiText = '';
      let aiIndex = null;
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        const chunk = decoder.decode(value || new Uint8Array(), { stream: !done });
        aiText += chunk;
        setMessages(m => {
          const arr = [...m];
          if (aiIndex === null) {
            aiIndex = arr.length;
            arr.push({ author: 'ai', text: aiText, timestamp: new Date().toISOString() });
          } else {
            arr[aiIndex] = { ...arr[aiIndex], text: aiText };
          }
          return arr;
        });
      }

      const [answer, sourceBlock] = aiText.split(/--- Sources ---/);
      const sources = sourceBlock ? sourceBlock.trim().split(/\n+/).filter(Boolean) : [];
      setMessages(m => {
        const arr = [...m];
        const msg = arr[aiIndex];
        arr[aiIndex] = { ...msg, text: answer.trim(), sources };
        return arr;
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen p-4">
      <div ref={containerRef} className="flex-1 overflow-y-auto space-y-4 p-4 border rounded">
        {messages.map((msg, idx) => (
          <div key={idx} className={msg.author === 'user' ? 'text-right' : 'text-left'}>
            <div className={`inline-block p-3 rounded-lg max-w-2xl ${msg.author === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}>
              <p className="whitespace-pre-wrap">{msg.text}</p>
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-2 text-sm text-gray-600">
                  <div className="font-semibold">Sources:</div>
                  <ul className="list-disc list-inside">
                    {msg.sources.map((src, i) => (
                      <li key={i}>{src}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {new Date(msg.timestamp).toLocaleString()}
            </div>
          </div>
        ))}
        {loading && <div className="text-gray-500">Loading...</div>}
        {error && <div className="text-red-600">{error}</div>}
      </div>
      <form onSubmit={e => { e.preventDefault(); sendMessage(); }} className="mt-4 flex">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          className="flex-1 border rounded p-2 mr-2 resize-none"
          placeholder="Type your question..."
          rows={2}
        />
        <button type="submit" disabled={loading} className="bg-blue-600 text-white px-4 py-2 rounded">
          Send
        </button>
      </form>
    </div>
  );
}


// Main Application JavaScript

// DOM Elements
const chatForm = document.getElementById('chatForm');
const queryInput = document.getElementById('queryInput');
const chatMessages = document.getElementById('chatMessages');
const sendButton = document.getElementById('sendButton');
const loadingOverlay = document.getElementById('loadingOverlay');
const statusBadge = document.getElementById('statusBadge');
const docCount = document.getElementById('docCount');
const modelName = document.getElementById('modelName');
const sourcesList = document.getElementById('sourcesList');

// State
let conversationHistory = [];
let allSources = new Set();

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    setupEventListeners();
    autoResizeTextarea();
    initTheme();
});

// Event Listeners
function setupEventListeners() {
    // Form submission
    chatForm.addEventListener('submit', handleSubmit);
    
    // Auto-resize textarea
    queryInput.addEventListener('input', autoResizeTextarea);
    
    // Enter key handling
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });
}

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();
    
    const query = queryInput.value.trim();
    if (!query) return;
    
    // Add user message
    addMessage(query, 'user');
    
    // Clear input
    queryInput.value = '';
    autoResizeTextarea();
    
    // Disable input
    setInputState(false);
    
    // Show loading
    showLoading(true);
    
    try {
        // Send request
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query }),
        });
        
        if (!response.ok) {
            throw new Error('Failed to get response');
        }
        
        const data = await response.json();
        
        // Add bot response with stats
        addMessage(data.answer, 'bot', data.sources, data.chunks, data.stats);
        
        // Update sources
        updateSources(data.sources);
        
    } catch (error) {
        console.error('Error:', error);
        addMessage(
            'Sorry, I encountered an error processing your request. Please try again.',
            'bot'
        );
    } finally {
        showLoading(false);
        setInputState(true);
        queryInput.focus();
    }
}

// Add message to chat (no benchmark stats or retrieved context)
function addMessage(text, type, sources = [], chunks = [], stats = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;

    const avatar = document.createElement('div');
    avatar.className = `message-avatar ${type}-avatar`;
    if (type === 'bot') {
        avatar.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L17 7V17L12 22L7 17V7L12 2Z" stroke="currentColor" stroke-width="2"/>
                <circle cx="12" cy="12" r="2" fill="currentColor"/>
            </svg>
        `;
    } else {
        avatar.textContent = 'You';
    }

    const content = document.createElement('div');
    content.className = 'message-content';
    const messageText = document.createElement('div');
    messageText.className = 'message-text';
    messageText.innerHTML = formatStructuredText(text);
    content.appendChild(messageText);

    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        const sourcesTitle = document.createElement('div');
        sourcesTitle.className = 'sources-title';
        sourcesTitle.innerHTML = `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2 3.5C2 2.67 2.67 2 3.5 2H12.5C13.33 2 14 2.67 14 3.5V12.5C14 13.33 13.33 14 12.5 14H3.5C2.67 14 2 13.33 2 12.5V3.5Z" stroke="currentColor" stroke-width="1.5"/></svg> Sources:`;
        const tags = document.createElement('div');
        sources.forEach(s => {
            const tag = document.createElement('span');
            tag.className = 'source-tag';
            tag.textContent = s;
            tags.appendChild(tag);
        });
        sourcesDiv.appendChild(sourcesTitle);
        sourcesDiv.appendChild(tags);
        content.appendChild(sourcesDiv);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    conversationHistory.push({ text, type, sources, chunks, stats });
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// Format bot output: ## headers, tables, lists, bold, paragraphs
function formatStructuredText(text) {
    if (!text) return '';
    const escaped = escapeHtml(text);
    const out = escaped
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>');
    const blocks = out.split(/\n\n+/).map(b => b.trim()).filter(Boolean);
    const result = [];
    for (const block of blocks) {
        const lines = block.split(/\n/).map(l => l.trim()).filter(Boolean);
        if (lines.length === 0) continue;
        const first = lines[0];
        if (first.startsWith('## ')) {
            const title = first.replace(/^##\s*/, '');
            result.push('<h3 class="msg-heading">' + title + '</h3>');
            if (lines.length === 1) continue;
            const rest = lines.slice(1).join('\n');
            result.push(renderBlock(rest, lines.slice(1)));
        } else if (isMarkdownTable(lines)) {
            result.push(renderTable(lines));
        } else {
            result.push(renderBlock(block, lines));
        }
    }
    return result.length ? result.join('') : '<p>' + out.replace(/\n/g, '<br>') + '</p>';
}

function renderBlock(block, lines) {
    if (!lines || lines.length === 0) return '<p>' + block.replace(/\n/g, '<br>') + '</p>';
    const allNumbered = lines.every(l => /^\d+[.)]\s/.test(l));
    const allBullets = lines.every(l => /^[-*•]\s/.test(l));
    if (allNumbered) {
        return '<ol class="msg-list">' + lines.map(l => '<li>' + l.replace(/^\d+[.)]\s*/, '') + '</li>').join('') + '</ol>';
    }
    if (allBullets) {
        return '<ul class="msg-list">' + lines.map(l => '<li>' + l.replace(/^[-*•]\s*/, '') + '</li>').join('') + '</ul>';
    }
    return '<p>' + block.replace(/\n/g, '<br>') + '</p>';
}

function isMarkdownTable(lines) {
    if (lines.length < 2) return false;
    const hasPipe = lines[0].includes('|');
    const sep = lines[1];
    const isSep = /^\|[\s\-:]+\|/.test(sep) && sep.includes('|');
    return hasPipe && isSep;
}

function renderTable(lines) {
    const rows = [];
    let i = 0;
    if (lines[0].includes('|')) {
        const headCells = lines[0].split('|').map(c => c.trim()).filter(Boolean);
        rows.push('<thead><tr>' + headCells.map(c => '<th>' + c + '</th>').join('') + '</tr></thead>');
        i = 2;
    }
    const body = [];
    for (; i < lines.length; i++) {
        if (!lines[i].includes('|')) break;
        const cells = lines[i].split('|').map(c => c.trim()).filter(Boolean);
        body.push('<tr>' + cells.map(c => '<td>' + c + '</td>').join('') + '</tr>');
    }
    rows.push('<tbody>' + body.join('') + '</tbody>');
    return '<div class="msg-table-wrap"><table class="msg-table">' + rows.join('') + '</table></div>';
}

// Update sources list
function updateSources(newSources) {
    newSources.forEach(source => allSources.add(source));
    
    sourcesList.innerHTML = '';
    
    if (allSources.size === 0) {
        sourcesList.innerHTML = '<p class="empty-state">No sources yet. Ask a question to get started!</p>';
        return;
    }
    
    Array.from(allSources).forEach(source => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';
        sourceItem.textContent = source;
        sourcesList.appendChild(sourceItem);
    });
}

// Load stats
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        docCount.textContent = data.documents_indexed || '0';
        modelName.textContent = data.embedding_model || 'Loading...';
        
        // Update status
        const statusText = statusBadge.querySelector('.status-text');
        statusText.textContent = data.status === 'ready' ? 'Ready' : 'Initializing...';
        
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Show/hide loading overlay
function showLoading(show) {
    if (show) {
        loadingOverlay.classList.add('active');
    } else {
        loadingOverlay.classList.remove('active');
    }
}

// Set input state
function setInputState(enabled) {
    queryInput.disabled = !enabled;
    sendButton.disabled = !enabled;
    
    if (enabled) {
        sendButton.innerHTML = `
            <svg class="send-icon" width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M2 10L18 2L10 18L9 11L2 10Z" fill="currentColor"/>
            </svg>
            <span class="button-text">Send</span>
        `;
    } else {
        sendButton.innerHTML = `
            <svg class="send-icon" width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="10" cy="10" r="2" fill="currentColor">
                    <animate attributeName="opacity" from="1" to="0.3" dur="1s" repeatCount="indefinite"/>
                </circle>
            </svg>
            <span class="button-text">Sending...</span>
        `;
    }
}

// Auto-resize textarea
function autoResizeTextarea() {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 150) + 'px';
}

// Send suggestion
function sendSuggestion(text) {
    queryInput.value = text;
    chatForm.dispatchEvent(new Event('submit'));
}

// Make sendSuggestion globally available
window.sendSuggestion = sendSuggestion;
window.loadStats = loadStats;

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K to focus input
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        queryInput.focus();
    }
    
    // Escape to clear input
    if (e.key === 'Escape') {
        queryInput.value = '';
        autoResizeTextarea();
    }
});

// Check health periodically
setInterval(async () => {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        const statusDot = statusBadge.querySelector('.status-dot');
        const statusText = statusBadge.querySelector('.status-text');
        
        if (data.status === 'healthy' && data.initialized) {
            statusDot.style.background = 'var(--success)';
            statusText.textContent = 'Ready';
        } else {
            statusDot.style.background = 'var(--warning)';
            statusText.textContent = 'Initializing...';
        }
    } catch (error) {
        const statusDot = statusBadge.querySelector('.status-dot');
        const statusText = statusBadge.querySelector('.status-text');
        statusDot.style.background = 'var(--error)';
        statusText.textContent = 'Error';
    }
}, 5000);

// Theme (dark mode)
function initTheme() {
    const saved = localStorage.getItem('rag-theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (saved === 'dark' || (!saved && prefersDark)) document.body.classList.add('dark');
    updateThemeIcons();
    const btn = document.getElementById('themeToggle');
    if (btn) btn.addEventListener('click', () => {
        document.body.classList.toggle('dark');
        localStorage.setItem('rag-theme', document.body.classList.contains('dark') ? 'dark' : 'light');
        updateThemeIcons();
    });
}

function updateThemeIcons() {
    const isDark = document.body.classList.contains('dark');
    const moon = document.querySelector('.theme-toggle .icon-moon');
    const sun = document.querySelector('.theme-toggle .icon-sun');
    if (moon) moon.style.display = isDark ? 'none' : '';
    if (sun) sun.style.display = isDark ? '' : 'none';
}

// Console message
console.log('%c🤖 Offline RAG Bot', 'font-size: 20px; font-weight: bold; color: #0d9488;');
console.log('%cBuilt with ❤️ using open-source tools', 'font-size: 12px; color: #6b7280;');
console.log('%c- SentenceTransformers for embeddings', 'font-size: 11px; color: #9ca3af;');
console.log('%c- FAISS for vector search', 'font-size: 11px; color: #9ca3af;');
console.log('%c- Llama.cpp for LLM inference', 'font-size: 11px; color: #9ca3af;');
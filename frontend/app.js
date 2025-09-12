// app.js - Dental AI Frontend Logic - Complete Version

// Configuration
const API_BASE_URL = 'http://localhost:8000';
const TENANT_ID = '11111111-1111-1111-1111-111111111111';

// State
let isProcessing = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkAPIStatus();
    setInterval(checkAPIStatus, 30000); // Check every 30 seconds
    document.getElementById('userInput').focus();
    
    // Set up scroll listener for scroll-to-top button
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.addEventListener('scroll', handleScroll);
});

// Check API Status
async function checkAPIStatus() {
    const statusElement = document.getElementById('apiStatus');
    
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            statusElement.innerHTML = '<i class="fas fa-check-circle"></i> API Online';
            statusElement.className = 'api-status online';
        } else {
            statusElement.innerHTML = '<i class="fas fa-exclamation-circle"></i> API Degraded';
            statusElement.className = 'api-status offline';
        }
    } catch (error) {
        statusElement.innerHTML = '<i class="fas fa-times-circle"></i> API Offline';
        statusElement.className = 'api-status offline';
        console.error('API Status Check Error:', error);
    }
}

// Handle scroll for scroll-to-top button visibility
function handleScroll() {
    const chatMessages = document.getElementById('chatMessages');
    const scrollBtn = document.getElementById('scrollToTop');
    
    if (chatMessages.scrollTop > 200) {
        scrollBtn.style.display = "block";
    } else {
        scrollBtn.style.display = "none";
    }
}

// Scroll to top functionality
function scrollToTop() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Clear chat functionality
function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    
    // Keep only the welcome message (first child)
    while (chatMessages.children.length > 1) {
        chatMessages.removeChild(chatMessages.lastChild);
    }
    
    // Clear the details panels
    document.getElementById('responseDetails').innerHTML = '<p class="placeholder-text">Response details and citations will appear here</p>';
    document.getElementById('agentActivity').innerHTML = '<p class="placeholder-text">Agent trace will appear here</p>';
    
    // Focus back on input
    document.getElementById('userInput').focus();
}

// Handle Enter key press
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Send message
async function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    
    if (!message || isProcessing) return;

    const validation = validateInput(message);
    if (!validation.valid) {
        addMessage(validation.message, 'bot');
        input.value = '';
        return;
    }
    
    // Clear input and disable
    input.value = '';
    isProcessing = true;
    updateSendButton();
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Determine endpoint based on message content
    const isComplexTask = message.toLowerCase().includes('schedule') || 
                         message.toLowerCase().includes('appointment') ||
                         message.toLowerCase().includes('book');
    
    try {
        if (isComplexTask) {
            await sendAgentRequest(message);
        } else {
            await sendAskRequest(message);
        }
    } catch (error) {
        console.error('Send Message Error:', error);
        addMessage('Sorry, I encountered an error. Please try again.', 'bot', {
            error: error.message
        });
    } finally {
        isProcessing = false;
        updateSendButton();
        input.focus();
    }
}

// Send quick query
function sendQuickQuery(query) {
    document.getElementById('userInput').value = query;
    sendMessage();
}

// Update send button state
function updateSendButton() {
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = isProcessing;
    
    if (isProcessing) {
        sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    } else {
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
    }
}

// Send request to /ask endpoint
async function sendAskRequest(query) {
    const userRole = document.getElementById('userRole').value;
    const userId = document.getElementById('userId').value;
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    try {
        const response = await fetch(`${API_BASE_URL}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                tenant_id: TENANT_ID,
                user_id: userId,
                user_role: userRole
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        if (response.ok) {
            // Add bot response with citations
            addMessage(data.answer, 'bot', {
                citations: data.citations,
                metrics: data.metrics,
                confidence: data.confidence,
                trace_id: data.trace_id
            });
            
            // Update response details panel
            updateResponseDetails(data);
        } else {
            throw new Error(data.detail || 'API request failed');
        }
    } catch (error) {
        removeTypingIndicator(typingId);
        throw error;
    }
}

// Send request to /agent endpoint
async function sendAgentRequest(task) {
    const userRole = document.getElementById('userRole').value;
    const userId = document.getElementById('userId').value;
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    try {
        const response = await fetch(`${API_BASE_URL}/agent`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                task: task,
                tenant_id: TENANT_ID,
                user_id: userId,
                user_role: userRole,
                parameters: {}
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        if (response.ok) {
            // Add bot response
            const responseText = data.result.response || data.result.answer || JSON.stringify(data.result);
            addMessage(responseText, 'bot', {
                agent: data.result.agent,
                trace: data.agent_trace,
                total_tokens: data.total_tokens,
                total_cost: data.total_cost,
                trace_id: data.trace_id
            });
            
            // Update agent activity panel
            updateAgentActivity(data.agent_trace);
        } else {
            throw new Error(data.detail || 'API request failed');
        }
    } catch (error) {
        removeTypingIndicator(typingId);
        throw error;
    }
}

// Add message to chat
function addMessage(text, sender, metadata = {}) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    let citationsHtml = '';
    if (metadata.citations && metadata.citations.length > 0) {
        citationsHtml = '<div class="citations"><strong>Sources:</strong>';
        metadata.citations.forEach((citation, index) => {
            citationsHtml += `
                <div class="citation">
                    <div class="citation-header">
                        <span class="citation-title">[${index + 1}] ${citation.metadata?.title || citation.doc_id}</span>
                        <span class="citation-score">${(citation.relevance_score * 100).toFixed(0)}% match</span>
                    </div>
                    <div class="citation-text">${citation.text}</div>
                </div>
            `;
        });
        citationsHtml += '</div>';
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-${sender === 'user' ? 'user' : 'robot'}"></i>
        </div>
        <div class="message-content">
            <div class="message-bubble">
                ${text}
                ${citationsHtml}
            </div>
            <div class="message-time">${time}</div>
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Show typing indicator
function showTypingIndicator() {
    const messagesContainer = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    const id = 'typing-' + Date.now();
    typingDiv.id = id;
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="message-bubble">
                <i class="fas fa-ellipsis-h fa-fade"></i>
            </div>
        </div>
    `;
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return id;
}

// Remove typing indicator
function removeTypingIndicator(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}

// Update response details panel
function updateResponseDetails(data) {
    const detailsDiv = document.getElementById('responseDetails');
    
    let html = '<div class="response-metrics">';
    html += `<div class="metric-item"><span class="metric-label">Confidence:</span><span class="metric-value">${(data.confidence * 100).toFixed(0)}%</span></div>`;
    html += `<div class="metric-item"><span class="metric-label">Latency:</span><span class="metric-value">${data.metrics.latency_ms.toFixed(0)}ms</span></div>`;
    html += `<div class="metric-item"><span class="metric-label">Results:</span><span class="metric-value">${data.metrics.search_results_count}</span></div>`;
    html += `<div class="metric-item"><span class="metric-label">Model:</span><span class="metric-value">${data.metrics.model_used}</span></div>`;
    html += `<div class="metric-item"><span class="metric-label">Trace ID:</span><span class="metric-value" style="font-size: 0.7rem;">${data.trace_id}</span></div>`;
    html += '</div>';
    
    detailsDiv.innerHTML = html;
}

// Update agent activity panel
function updateAgentActivity(trace) {
    const activityDiv = document.getElementById('agentActivity');
    
    let html = '';
    if (trace && trace.length > 0) {
        trace.forEach(step => {
            html += `
                <div class="agent-step">
                    <div class="agent-name">${step.agent}</div>
                    <div class="agent-action">${step.action || step.task || 'Processing'}</div>
                </div>
            `;
        });
    }
    
    activityDiv.innerHTML = html || '<p class="placeholder-text">No agent activity</p>';
}

function validateInput(message) {
    // Check for inappropriate content
    const inappropriate = ['sex', 'porn', 'drug', 'weapon', 'hack'];
    const messageLower = message.toLowerCase();
    
    for (let word of inappropriate) {
        if (messageLower.includes(word)) {
            return {
                valid: false,
                message: "Please keep the conversation professional and focused on dental health topics."
            };
        }
    }
    
    // Check message length
    if (message.length > 500) {
        return {
            valid: false,
            message: "Please keep your question under 500 characters."
        };
    }
    
    return { valid: true };
}

// Show metrics modal
async function showMetrics() {
    const modal = document.getElementById('metricsModal');
    const content = document.getElementById('metricsContent');
    
    modal.style.display = 'block';
    content.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Loading metrics...</div>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/metrics`);
        const data = await response.json();
        
        let html = '<div class="metrics-grid">';
        
        // Request metrics
        html += `
            <div class="metric-card">
                <h4>Total Requests</h4>
                <div class="value">${data.requests.total}</div>
            </div>
            <div class="metric-card">
                <h4>Requests/Min</h4>
                <div class="value">${data.requests.per_minute}</div>
            </div>
            <div class="metric-card">
                <h4>Error Rate</h4>
                <div class="value">${(data.requests.error_rate * 100).toFixed(1)}%</div>
            </div>
        `;
        
        // Performance metrics
        html += `
            <div class="metric-card">
                <h4>Avg Latency</h4>
                <div class="value">${data.performance.avg_latency_ms}ms</div>
            </div>
            <div class="metric-card">
                <h4>P95 Latency</h4>
                <div class="value">${data.performance.p95_latency_ms}ms</div>
            </div>
            <div class="metric-card">
                <h4>P99 Latency</h4>
                <div class="value">${data.performance.p99_latency_ms}ms</div>
            </div>
        `;
        
        // RAG metrics
        html += `
            <div class="metric-card">
                <h4>Documents</h4>
                <div class="value">${data.rag.documents_count}</div>
            </div>
            <div class="metric-card">
                <h4>Avg Results</h4>
                <div class="value">${data.rag.avg_search_results}</div>
            </div>
            <div class="metric-card">
                <h4>Cache Hit Rate</h4>
                <div class="value">${(data.rag.cache_hit_rate * 100).toFixed(0)}%</div>
            </div>
        `;
        
        html += '</div>';
        content.innerHTML = html;
        
    } catch (error) {
        content.innerHTML = '<p style="color: var(--danger-color);">Failed to load metrics</p>';
    }
}

// Close metrics modal
function closeMetrics() {
    document.getElementById('metricsModal').style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('metricsModal');
    if (event.target == modal) {
        modal.style.display = 'none';
    }
}
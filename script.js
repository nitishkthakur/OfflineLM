function streamingChatApp() {
    return {
        messages: [],
        currentMessage: '',
        isLoading: false,
        messageId: 0,
        statusMessage: 'Ready to chat!',
        currentEventSource: null,
        currentStreamSession: null,
        retryAttempts: 0,
        maxRetries: 2,
        
        init() {
            // Configure marked.js with highlight.js
            marked.setOptions({
                highlight: function(code, lang) {
                    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                    return hljs.highlight(code, { language }).value;
                },
                breaks: true,
                gfm: true,
                tables: true,
                sanitize: false
            });
            
            // Make this instance globally accessible
            Alpine.store('chatApp', this);
            window.chatAppInstance = this;
            
            // Load existing messages on page load
            this.loadMessages();
        },
        
        async loadMessages() {
            try {
                const response = await fetch('/messages');
                const data = await response.json();
                this.messages = data.messages.map((msg, index) => ({
                    id: index,
                    type: msg.type,
                    content: this.formatMessage(msg.content),
                    streaming: false
                }));
                this.messageId = this.messages.length;
                this.$nextTick(() => this.scrollToBottom());
            } catch (error) {
                console.error('Error loading messages:', error);
                this.statusMessage = 'Error loading messages';
            }
        },
        
        formatMessage(content) {
            try {
                // Process think tags first before markdown parsing
                const processedContent = this.processThinkTags(content);
                let html = marked.parse(processedContent);
                setTimeout(() => {
                    document.querySelectorAll('pre code').forEach((block) => {
                        hljs.highlightElement(block);
                    });
                }, 0);
                return html;
            } catch (error) {
                console.error('Error parsing markdown:', error);
                return this.processThinkTags(content)
                    .replace(/\n/g, '<br>')
                    .replace(/  /g, '&nbsp;&nbsp;')
                    .replace(/\t/g, '&nbsp;&nbsp;&nbsp;&nbsp;');
            }
        },

        processThinkTags(content) {
            // Check if content starts with <think> and contains </think>
            const thinkRegex = /^(<think>)([\s\S]*?)(<\/think>)([\s\S]*)$/;
            const match = content.match(thinkRegex);
            
            if (match) {
                const [, openTag, thinkContent, closeTag, remainingContent] = match;
                
                // Clean up the think content - preserve newlines and structure
                const cleanThinkContent = thinkContent.trim();
                
                // Format the think section with special styling
                const formattedThinkSection = `<div class="think-section"><em>${openTag}\n${cleanThinkContent}\n${closeTag}</em></div>`;
                
                // Add proper spacing after think section
                if (remainingContent.trim()) {
                    return formattedThinkSection + '\n\n' + remainingContent.trim();
                } else {
                    return formattedThinkSection;
                }
            }
            
            // If we're in the middle of streaming and see <think> at the start but no closing tag yet,
            // we'll format it differently to show it's in progress
            if (content.startsWith('<think>') && !content.includes('</think>')) {
                // Format as in-progress think section
                return `<div class="think-section think-streaming"><em>${content}</em></div>`;
            }
            
            return content;
        },

        cleanContentForBackend(content) {
            // This function returns clean content suitable for backend processing
            // It preserves think tags in their original form for proper PDF generation
            return content; // Raw content is already clean - no HTML processing needed
        },
        
        async sendMessage() {
            if (!this.currentMessage.trim() || this.isLoading) return;
            
            const userMessage = this.currentMessage;
            this.currentMessage = '';
            
            // Close any existing EventSource
            if (this.currentEventSource) {
                this.currentEventSource.close();
                this.currentEventSource = null;
            }
            
            // Add user message
            this.addMessage('user', userMessage, false);
            
            // Start streaming response
            this.isLoading = true;
            this.statusMessage = 'Sending message...';
            
            // Add empty assistant message for streaming
            const assistantMessageId = this.messageId;
            this.addMessage('assistant', '', true);
            
            try {
                // Generate unique session ID for this stream
                const sessionId = 'stream_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                this.currentStreamSession = sessionId;
                
                // Create EventSource for streaming
                // Get current model with proper fallback chain
                let currentModel = Alpine.store('configPanel')?.selectedModel || 
                                   window.configPanelInstance?.selectedModel;
                
                // If no model is set, get default from server
                if (!currentModel) {
                    try {
                        const defaultResponse = await fetch('/default-model');
                        const defaultData = await defaultResponse.json();
                        currentModel = defaultData.default_model || 'qwen2.5:7b';
                    } catch (e) {
                        currentModel = 'qwen2.5:7b'; // Ultimate fallback
                    }
                }
                console.log('Using model for streaming:', currentModel);
                const eventSource = new EventSource(`/chat/stream-sse?message=${encodeURIComponent(userMessage)}&model=${encodeURIComponent(currentModel)}&session=${sessionId}`);
                this.currentEventSource = eventSource;
                
                // Add connection state tracking
                let connectionEstablished = false;
                let hasReceivedData = false;
                
                eventSource.onopen = () => {
                    connectionEstablished = true;
                    this.statusMessage = 'Connected, waiting for response...';
                };
                
                eventSource.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        
                        if (data.type === 'chunk' && data.content) {
                            hasReceivedData = true;
                            // Find the assistant message and append content
                            const assistantMsg = this.messages.find(m => m.id === assistantMessageId);
                            if (assistantMsg) {
                                // Append raw content first, then format
                                const currentRawContent = assistantMsg.rawContent || '';
                                assistantMsg.rawContent = currentRawContent + data.content;
                                assistantMsg.content = this.formatMessage(assistantMsg.rawContent);
                            }
                            this.statusMessage = 'Streaming response...';
                            this.$nextTick(() => this.scrollToBottom());
                            
                        } else if (data.type === 'done') {
                            const assistantMsg = this.messages.find(m => m.id === assistantMessageId);
                            if (assistantMsg) {
                                assistantMsg.streaming = false;
                            }
                            eventSource.close();
                            this.currentEventSource = null;
                            this.isLoading = false;
                            this.statusMessage = 'Response complete';
                            setTimeout(() => this.statusMessage = 'Ready to chat!', 2000);
                            
                        } else if (data.type === 'stopped') {
                            const assistantMsg = this.messages.find(m => m.id === assistantMessageId);
                            if (assistantMsg) {
                                assistantMsg.streaming = false;
                                assistantMsg.content += '\n\n*[Stream stopped]*';
                            }
                            eventSource.close();
                            this.currentEventSource = null;
                            this.isLoading = false;
                            this.statusMessage = 'Stream stopped';
                            setTimeout(() => this.statusMessage = 'Ready to chat!', 2000);
                            
                            // Return focus to the input box after streaming is complete
                            this.$nextTick(() => {
                                const inputElement = document.querySelector('.chat-input');
                                if (inputElement) {
                                    inputElement.focus();
                                }
                            });
                            
                        } else if (data.type === 'error') {
                            const assistantMsg = this.messages.find(m => m.id === assistantMessageId);
                            if (assistantMsg) {
                                // Preserve existing content and append error
                                const existingContent = assistantMsg.rawContent || assistantMsg.content.replace(/<[^>]*>/g, '');
                                if (existingContent.trim()) {
                                    assistantMsg.rawContent = existingContent + '\n\n*[Error: ' + data.message + ']*';
                                    assistantMsg.content = this.formatMessage(assistantMsg.rawContent);
                                } else {
                                    assistantMsg.content = '❌ Error: ' + data.message;
                                    assistantMsg.rawContent = '❌ Error: ' + data.message;
                                }
                                assistantMsg.streaming = false;
                            }
                            eventSource.close();
                            this.currentEventSource = null;
                            this.isLoading = false;
                            this.statusMessage = 'Error occurred';
                            
                            // Return focus to input box on error
                            this.$nextTick(() => {
                                const inputElement = document.querySelector('.chat-input');
                                if (inputElement) {
                                    inputElement.focus();
                                }
                            });
                        }
                    } catch (e) {
                        console.error('Error parsing SSE data:', e);
                    }
                };
                
                eventSource.onerror = (event) => {
                    console.error('EventSource error:', event, 'ReadyState:', eventSource.readyState);
                    const assistantMsg = this.messages.find(m => m.id === assistantMessageId);
                    if (assistantMsg) {
                        // Preserve existing content and append error with more context
                        const existingContent = assistantMsg.rawContent || assistantMsg.content.replace(/<[^>]*>/g, '');
                        let errorMessage = '';
                        
                        if (!connectionEstablished) {
                            errorMessage = '*[Failed to connect to server - please check connection]*';
                        } else if (!hasReceivedData) {
                            errorMessage = '*[Connection lost before receiving response - server may be overloaded]*';
                        } else {
                            errorMessage = '*[Connection interrupted - partial response received]*';
                        }
                        
                        if (existingContent.trim()) {
                            assistantMsg.rawContent = existingContent + '\n\n' + errorMessage;
                            assistantMsg.content = this.formatMessage(assistantMsg.rawContent);
                        } else {
                            assistantMsg.content = '❌ Connection error. Please try again.';
                            assistantMsg.rawContent = '❌ Connection error. Please try again.';
                        }
                        assistantMsg.streaming = false;
                    }
                    eventSource.close();
                    this.currentEventSource = null;
                    this.isLoading = false;
                    this.statusMessage = 'Connection error';
                    
                    // Return focus to input box on connection error
                    this.$nextTick(() => {
                        const inputElement = document.querySelector('.chat-input');
                        if (inputElement) {
                            inputElement.focus();
                        }
                    });
                };
                
                // Timeout fallback
                setTimeout(() => {
                    if (this.currentEventSource && this.currentEventSource.readyState === EventSource.CONNECTING) {
                        eventSource.close();
                        const assistantMsg = this.messages.find(m => m.id === assistantMessageId);
                        if (assistantMsg) {
                            // Preserve existing content and append timeout error
                            const existingContent = assistantMsg.rawContent || assistantMsg.content.replace(/<[^>]*>/g, '');
                            if (existingContent.trim()) {
                                assistantMsg.rawContent = existingContent + '\n\n*[Request timeout - response incomplete]*';
                                assistantMsg.content = this.formatMessage(assistantMsg.rawContent);
                            } else {
                                assistantMsg.content = '⏱️ Request timeout. Please try again.';
                                assistantMsg.rawContent = '⏱️ Request timeout. Please try again.';
                            }
                            assistantMsg.streaming = false;
                        }
                        this.currentEventSource = null;
                        this.isLoading = false;
                        this.statusMessage = 'Request timeout';
                        
                        // Return focus to input box on timeout
                        this.$nextTick(() => {
                            const inputElement = document.querySelector('.chat-input');
                            if (inputElement) {
                                inputElement.focus();
                            }
                        });
                    }
                }, 60000); // 60 second timeout
                
            } catch (error) {
                console.error('Error setting up streaming:', error);
                const assistantMsg = this.messages.find(m => m.id === assistantMessageId);
                if (assistantMsg) {
                    // For setup errors, there shouldn't be existing content, but check anyway
                    const existingContent = assistantMsg.rawContent || assistantMsg.content.replace(/<[^>]*>/g, '');
                    if (existingContent.trim()) {
                        assistantMsg.rawContent = existingContent + '\n\n*[Streaming setup failed]*';
                        assistantMsg.content = this.formatMessage(assistantMsg.rawContent);
                    } else {
                        assistantMsg.content = '❌ Failed to start streaming. Please try again.';
                        assistantMsg.rawContent = '❌ Failed to start streaming. Please try again.';
                    }
                    assistantMsg.streaming = false;
                }
                this.isLoading = false;
                this.statusMessage = 'Failed to start streaming';
                
                // Return focus to input box on setup error
                this.$nextTick(() => {
                    const inputElement = document.querySelector('.chat-input');
                    if (inputElement) {
                        inputElement.focus();
                    }
                });
            }
        },
        
        stopStreaming() {
            if (this.currentEventSource) {
                console.log('Stopping stream...');
                
                // Send stop request to backend if we have a session
                if (this.currentStreamSession) {
                    fetch(`/stop-stream/${this.currentStreamSession}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    }).then(response => {
                        console.log('Stop request sent:', response.status);
                    }).catch(error => {
                        console.error('Error sending stop request:', error);
                    });
                }
                
                // Close EventSource connection
                this.currentEventSource.close();
                
                // Clean up state
                this.cleanupStream();
                
                // Update any streaming message
                const streamingMsg = this.messages.find(msg => msg.streaming);
                if (streamingMsg) {
                    streamingMsg.streaming = false;
                    // Preserve content properly
                    const existingContent = streamingMsg.rawContent || streamingMsg.content.replace(/<[^>]*>/g, '');
                    streamingMsg.rawContent = existingContent + '\n\n*[Streaming stopped by user]*';
                    streamingMsg.content = this.formatMessage(streamingMsg.rawContent);
                }
                
                this.statusMessage = 'Streaming stopped';
                setTimeout(() => this.statusMessage = 'Ready to chat!', 2000);
                
                // Return focus to input
                this.$nextTick(() => {
                    const inputElement = document.querySelector('.chat-input');
                    if (inputElement) {
                        inputElement.focus();
                    }
                });
            }
        },
        
        cleanupStream() {
            if (this.currentEventSource) {
                this.currentEventSource.close();
                this.currentEventSource = null;
            }
            this.currentStreamSession = null;
            this.isLoading = false;
        },
        
        addMessage(type, content, streaming = false) {
            this.messages.push({
                id: this.messageId++,
                type: type,
                content: this.formatMessage(content),
                rawContent: content,
                streaming: streaming
            });
            this.$nextTick(() => this.scrollToBottom());
        },
        
        async clearChat() {
            try {
                await fetch('/clear', { method: 'POST' });
                this.messages = [];
                this.messageId = 0;
                this.statusMessage = 'Chat cleared';
                setTimeout(() => this.statusMessage = 'Ready to chat!', 2000);
            } catch (error) {
                console.error('Error clearing chat:', error);
                this.statusMessage = 'Error clearing chat';
            }
        },
        
        scrollToBottom() {
            const chatMessages = this.$refs.chatMessages;
            chatMessages.scrollTop = chatMessages.scrollHeight;
        },
        
        // Expose method to change model from config panel
        changeModel(newModel) {
            this.statusMessage = `Switching to ${newModel}...`;
            // Model change is handled by the config panel
            setTimeout(() => this.statusMessage = 'Ready to chat!', 2000);
        }
    }
}

function configPanel() {
    return {
        availableModels: [],
        selectedModel: null, // Will be set dynamically
        selectedModelInfo: null,
        isLoadingModels: false,
        chatStats: {
            messageCount: 0,
            currentModel: null, // Will be set dynamically
            status: 'Ready'
        },
        modelStatus: [],
        modelStatusMessages: [],
        
        init() {
            // Store this instance globally for access from chat app
            Alpine.store('configPanel', this);
            window.configPanelInstance = this;
            
            this.loadAvailableModels();
            this.updateStats();
        },
        
        async loadAvailableModels() {
            this.isLoadingModels = true;
            try {
                const response = await fetch('/models');
                const data = await response.json();
                this.availableModels = data.models || [];
                
                // Use the server-provided default model or first available model
                const defaultModel = data.default_model || (this.availableModels.length > 0 ? this.availableModels[0].name : 'qwen2.5:7b');
                
                // Set the default model if not already set
                if (!this.selectedModel) {
                    this.selectedModel = defaultModel;
                    this.chatStats.currentModel = defaultModel;
                }
                
                // Set selected model info
                this.selectedModelInfo = this.availableModels.find(model => model.name === this.selectedModel);
                
                console.log(`Default model set to: ${this.selectedModel}`);
                
            } catch (error) {
                console.error('Error loading models:', error);
                
                // Fallback: get default from server or use hardcoded fallback
                try {
                    const defaultResponse = await fetch('/default-model');
                    const defaultData = await defaultResponse.json();
                    const fallbackDefault = defaultData.default_model || 'qwen2.5:7b';
                    
                    // Fallback models if API fails
                    this.availableModels = [
                        { name: fallbackDefault, size: 'Unknown', modified_at: 'Recently' },
                        { name: 'qwen2.5:7b', size: '4.7GB', modified_at: 'Recently' },
                        { name: 'gemma3:4b-it-fp16', size: '2.4GB', modified_at: 'Recently' },
                        { name: 'llama3.2:3b', size: '2.0GB', modified_at: 'Recently' }
                    ];
                    
                    this.selectedModel = fallbackDefault;
                    this.chatStats.currentModel = fallbackDefault;
                    
                } catch (defaultError) {
                    console.error('Error getting default model:', defaultError);
                    // Final fallback
                    const ultimateFallback = 'qwen2.5:7b';
                    this.availableModels = [
                        { name: ultimateFallback, size: 'Unknown', modified_at: 'Recently' }
                    ];
                    this.selectedModel = ultimateFallback;
                    this.chatStats.currentModel = ultimateFallback;
                }
                
                this.selectedModelInfo = this.availableModels[0];
            } finally {
                this.isLoadingModels = false;
            }
        },
        
        async changeModel() {
            console.log('changeModel called with:', this.selectedModel);
            try {
                // Update selected model info
                this.selectedModelInfo = this.availableModels.find(model => model.name === this.selectedModel);
                this.chatStats.currentModel = this.selectedModel;
                this.chatStats.status = 'Switching Model...';
                
                // Add status message about model switching
                this.addModelStatusMessage(`🔄 Switching to ${this.selectedModel}...`);
                
                console.log('Attempting to change model to:', this.selectedModel);
                
                // Send model change to backend with conversation history
                const chatApp = Alpine.store('chatApp') || window.chatAppInstance;
                if (chatApp) {
                    const conversationHistory = chatApp.messages.map(msg => ({
                        role: msg.type === 'user' ? 'user' : 'assistant',
                        content: msg.rawContent || chatApp.cleanContentForBackend(msg.content.replace(/<[^>]*>/g, ''))
                    }));
                    
                    console.log('Sending model change request with history length:', conversationHistory.length);
                    
                    const response = await fetch('/change-model', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            model: this.selectedModel,
                            conversation_history: conversationHistory
                        })
                    });
                    
                    const result = await response.json();
                    console.log('Model change response:', result);
                    
                    // Handle stopped models
                    if (result.stopped_models && result.stopped_models.length > 0) {
                        result.stopped_models.forEach(modelName => {
                            this.addModelStatusMessage(`⏹️ Stopped model: ${modelName}`);
                        });
                    }
                    
                    if (result.status === 'success') {
                        this.addModelStatusMessage(`✅ ${this.selectedModel} loaded with 30min keepalive`);
                        this.chatStats.status = 'Model Changed';
                    } else {
                        this.addModelStatusMessage(`❌ Failed to switch: ${result.message}`);
                        this.chatStats.status = 'Error';
                    }
                    
                    // Notify chat app about model change
                    if (chatApp.changeModel) {
                        chatApp.changeModel(this.selectedModel);
                    }
                } else {
                    console.error('Chat app not found');
                    this.addModelStatusMessage('❌ Chat app not found');
                }
                
                // Update model status
                await this.updateModelStatus();
                
                setTimeout(() => this.chatStats.status = 'Ready', 3000);
            } catch (error) {
                console.error('Error changing model:', error);
                this.chatStats.status = 'Error';
                this.addModelStatusMessage(`❌ Error: ${error.message}`);
            }
        },
        
        addModelStatusMessage(message) {
            const timestamp = new Date().toLocaleTimeString();
            this.modelStatusMessages.unshift({
                message: message,
                timestamp: timestamp,
                id: Date.now()
            });
            
            // Keep only the last 10 messages
            if (this.modelStatusMessages.length > 10) {
                this.modelStatusMessages = this.modelStatusMessages.slice(0, 10);
            }
        },
        
        async updateModelStatus() {
            try {
                const response = await fetch('/model-status');
                const result = await response.json();
                
                if (result.status === 'success') {
                    this.modelStatus = result.models;
                }
            } catch (error) {
                console.error('Error updating model status:', error);
            }
        },
        
        async downloadChatAsPDF() {
            const chatApp = Alpine.store('chatApp') || window.chatAppInstance;
            if (!chatApp || chatApp.messages.length === 0) {
                alert('No chat messages to download.');
                return;
            }
            
            try {
                console.log('Starting PDF download...');
                
                // Make API call to backend
                const response = await fetch('/download-chat-pdf', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                console.log('Response status:', response.status);
                
                if (response.ok) {
                    // Get the PDF blob from response
                    const blob = await response.blob();
                    console.log('Blob size:', blob.size);
                    
                    // Create download link
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    
                    // Get filename from response headers or use default
                    const contentDisposition = response.headers.get('content-disposition');
                    let filename = 'chat-export.pdf';
                    if (contentDisposition) {
                        const filenameMatch = contentDisposition.match(/filename="(.+)"/);
                        if (filenameMatch) {
                            filename = filenameMatch[1];
                        }
                    }
                    
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                    
                    console.log(`Chat exported as ${filename}`);
                } else {
                    const errorText = await response.text();
                    console.error('PDF generation failed:', errorText);
                    throw new Error(`Server error: ${response.status} - ${errorText}`);
                }
                
            } catch (error) {
                console.error('Error downloading PDF:', error);
                alert('Error generating PDF: ' + error.message);
            }
        },
        
        updateStats() {
            // Update stats and model status periodically
            setInterval(() => {
                const chatApp = Alpine.store('chatApp') || window.chatAppInstance;
                if (chatApp) {
                    this.chatStats.messageCount = chatApp.messages.length;
                    if (chatApp.isLoading) {
                        this.chatStats.status = 'Processing...';
                    } else if (this.chatStats.status === 'Processing...') {
                        this.chatStats.status = 'Ready';
                    }
                }
            }, 1000);
            
            // Update model status every 30 seconds
            setInterval(() => {
                this.updateModelStatus();
            }, 30000);
            
            // Initial model status update
            this.updateModelStatus();
        }
    }
}

// Mobile config toggle
function toggleConfig() {
    const sidebar = document.querySelector('.config-sidebar');
    if (sidebar) {
        sidebar.style.display = sidebar.style.display === 'none' ? 'flex' : 'none';
    }
}

// Make chat app globally accessible for config panel
document.addEventListener('alpine:init', () => {
    Alpine.store('chatApp', null);
});

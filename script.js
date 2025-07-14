function streamingChatApp() {
    return {
        messages: [],
        currentMessage: '',
        isLoading: false,
        messageId: 0,
        statusMessage: 'Ready to chat!',
        currentEventSource: null,
        
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
                let html = marked.parse(content);
                setTimeout(() => {
                    document.querySelectorAll('pre code').forEach((block) => {
                        hljs.highlightElement(block);
                    });
                }, 0);
                return html;
            } catch (error) {
                console.error('Error parsing markdown:', error);
                return content
                    .replace(/\n/g, '<br>')
                    .replace(/  /g, '&nbsp;&nbsp;')
                    .replace(/\t/g, '&nbsp;&nbsp;&nbsp;&nbsp;');
            }
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
                // Create EventSource for streaming
                const currentModel = Alpine.store('configPanel')?.selectedModel || 
                                   window.configPanelInstance?.selectedModel || 
                                   'qwen3:4b';
                console.log('Using model for streaming:', currentModel);
                const eventSource = new EventSource(`/chat/stream-sse?message=${encodeURIComponent(userMessage)}&model=${encodeURIComponent(currentModel)}`);
                this.currentEventSource = eventSource;
                
                eventSource.onopen = () => {
                    this.statusMessage = 'Connected, waiting for response...';
                };
                
                eventSource.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        
                        if (data.type === 'chunk' && data.content) {
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
                                assistantMsg.content = '❌ Error: ' + data.message;
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
                    console.error('EventSource error:', event);
                    const assistantMsg = this.messages.find(m => m.id === assistantMessageId);
                    if (assistantMsg) {
                        assistantMsg.content = '❌ Connection error. Please try again.';
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
                            assistantMsg.content = '⏱️ Request timeout. Please try again.';
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
                    assistantMsg.content = '❌ Failed to start streaming. Please try again.';
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
        selectedModel: 'qwen3:4b', // Default model
        selectedModelInfo: null,
        isLoadingModels: false,
        chatStats: {
            messageCount: 0,
            currentModel: 'qwen3:4b',
            status: 'Ready'
        },
        
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
                
                // Set selected model info
                this.selectedModelInfo = this.availableModels.find(model => model.name === this.selectedModel);
                this.chatStats.currentModel = this.selectedModel;
            } catch (error) {
                console.error('Error loading models:', error);
                // Fallback models if API fails
                this.availableModels = [
                    { name: 'qwen3:4b', size: '2.6GB', modified_at: 'Recently' },
                    { name: 'gemma3:4b', size: '3.3GB', modified_at: 'Recently' },
                    { name: 'llama3.2:1b', size: '1.3GB', modified_at: 'Recently' },
                    { name: 'deepseek-r1:8b', size: '5.2GB', modified_at: 'Recently' }
                ];
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
                this.chatStats.status = 'Model Changing...';
                
                console.log('Attempting to change model to:', this.selectedModel);
                
                // Send model change to backend with conversation history
                const chatApp = Alpine.store('chatApp') || window.chatAppInstance;
                if (chatApp) {
                    const conversationHistory = chatApp.messages.map(msg => ({
                        role: msg.type === 'user' ? 'user' : 'assistant',
                        content: msg.rawContent || msg.content.replace(/<[^>]*>/g, '') // Strip HTML for raw content
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
                    
                    // Notify chat app about model change
                    if (chatApp.changeModel) {
                        chatApp.changeModel(this.selectedModel);
                    }
                } else {
                    console.error('Chat app not found');
                }
                
                this.chatStats.status = 'Model Changed';
                setTimeout(() => this.chatStats.status = 'Ready', 2000);
            } catch (error) {
                console.error('Error changing model:', error);
                this.chatStats.status = 'Error';
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
            // Update stats periodically
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

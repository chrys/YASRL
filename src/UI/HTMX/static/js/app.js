/**
 * HTMX SPA Application Script
 * Handles page transitions and active state management
 */

// Get base path from HTML data attribute (set by Flask's request.script_root)
const basePath = document.documentElement.getAttribute('data-base-path') || '';

// Export to global scope for use in dynamically loaded pages
window.basePath = basePath;

// Helper function to build URLs with base path
function buildUrl(path) {
    // If path already starts with base path, return as is
    if (basePath && path.startsWith(basePath)) {
        return path;
    }
    // If basePath is set and path doesn't start with /, add it
    if (basePath && !path.startsWith('/')) {
        return basePath + '/' + path;
    }
    // Otherwise return path as is
    return path;
}

// Export to global scope for use in dynamically loaded pages
window.buildUrl = buildUrl;

document.addEventListener('DOMContentLoaded', function() {
    console.log('[APP] DOMContentLoaded, basePath:', basePath);
    initializeApp();
});

function initializeApp() {
    console.log('[APP] Initializing app...');
    // Load admin page by default (use relative URL)
    loadPage('page/admin', null);
}

/**
 * Load a page dynamically
 */
function loadPage(url, event) {
    if (event) {
        event.preventDefault();
    }
    
    console.log('[APP] Loading page:', url);
    
    fetch(url)
        .then(response => {
            console.log('[APP] Response status:', response.status);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.text();
        })
        .then(html => {
            console.log('[APP] Received HTML, length:', html.length);
            const contentDiv = document.getElementById('page-content');
            if (!contentDiv) {
                throw new Error('Content container not found');
            }
            contentDiv.innerHTML = html;
            
            // IMPORTANT: Process HTMX attributes in the newly loaded HTML
            console.log('[APP] Processing HTMX attributes...');
            htmx.process(contentDiv);
            
            // Update active link
            updateActiveLink(url);
            
            // Execute any scripts in the injected HTML
            const scripts = contentDiv.querySelectorAll('script');
            scripts.forEach(script => {
                const newScript = document.createElement('script');
                newScript.textContent = script.textContent;
                document.head.appendChild(newScript);
            });
            
            // Dispatch a custom event so pages can hook into page load
            const event = new CustomEvent('pageLoaded', { detail: { url: url } });
            document.dispatchEvent(event);
            
            console.log('[APP] Page loaded and rendered');
        })
        .catch(error => {
            console.error('[APP] Error loading page:', error);
            const contentDiv = document.getElementById('page-content');
            if (contentDiv) {
                contentDiv.innerHTML = 
                    `<div class="card"><p style="color: red;">Error loading page: ${error.message}</p></div>`;
            }
        });
}

/**
 * Update the active navigation link
 */
function updateActiveLink(url) {
    const links = document.querySelectorAll('.nav-link');
    links.forEach(link => {
        link.classList.remove('active');
        // Match the href with the URL
        const href = link.getAttribute('href');
        if (url && url.includes(href.replace('#', ''))) {
            link.classList.add('active');
        }
    });
}

/**
 * Handle page transitions
 */
function handlePageTransition(event) {
    if (event.detail.xhr.status === 200) {
        // Update active link after successful request
        updateActiveLink();
    }
}

/**
 * Handle HTMX history navigation (back/forward buttons)
 */
document.addEventListener('htmx:historyRestore', function(event) {
    updateActiveLink();
});

function addSource(projectId, sourceType) {
    console.log(`[addSource] Called with projectId=${projectId}, sourceType=${sourceType}`);
    let source = null;
    
    if (sourceType === 'text') {
        const input = document.getElementById(`source-text-input-${projectId}`);
        console.log(`[addSource] Looking for input: source-text-input-${projectId}`, input);
        source = input.value.trim();
        if (!source) {
            alert('Please enter a source path or URL');
            return;
        }
    } else if (sourceType === 'file') {
        const fileInput = document.getElementById(`source-file-input-${projectId}`);
        console.log(`[addSource] Looking for file input: source-file-input-${projectId}`, fileInput);
        if (!fileInput.files.length) {
            alert('Please select a file');
            return;
        }
        
        const formData = new FormData();
        formData.append('project_id', projectId);
        formData.append('source_type', 'file');
        formData.append('file', fileInput.files[0]);
        
        console.log(`[addSource] Starting file upload`);
        uploadSourceAndIndex(projectId, formData);
        return;
    }
    
    const formData = new FormData();
    formData.append('project_id', projectId);
    formData.append('source_type', 'text');
    formData.append('source', source);
    
    console.log(`[addSource] Starting text source upload`);
    uploadSourceAndIndex(projectId, formData);
}

function uploadSourceAndIndex(projectId, formData) {
    console.log(`[uploadSourceAndIndex] Starting for projectId=${projectId}`);
    
    try {
        const indexingContainer = document.getElementById(`indexing-container-${projectId}`);
        const statusMessage = document.getElementById(`status-message-${projectId}`);
        const progressFill = document.getElementById(`progress-fill-${projectId}`);
        const progressText = document.getElementById(`progress-text-${projectId}`);
        const indexingStatus = document.getElementById(`indexing-status-${projectId}`);
        const indexingErrors = document.getElementById(`indexing-errors-${projectId}`);
        
        console.log(`[uploadSourceAndIndex] Container:`, indexingContainer, statusMessage, progressFill);
        
        if (!indexingContainer) {
            console.error(`[uploadSourceAndIndex] Could not find indexing-container-${projectId}`);
            alert('Error: Progress container not found. Please refresh and try again.');
            return;
        }
        
        // Show progress container
        indexingContainer.style.display = 'block';
        console.log(`[uploadSourceAndIndex] Set indexing container display to block`);
        if (statusMessage) statusMessage.style.display = 'none';
        if (indexingErrors) {
            indexingErrors.style.display = 'none';
            indexingErrors.innerHTML = '';
        }
        if (progressFill) progressFill.style.width = '0%';
        if (progressText) progressText.textContent = '0%';
        if (indexingStatus) indexingStatus.textContent = 'Adding source...';
        
        console.log(`[uploadSourceAndIndex] Sending request to api/projects/sources/add`);
        
        fetch('api/projects/sources/add', { method: 'POST', body: formData })
            .then(response => {
                console.log(`[uploadSourceAndIndex] Response received:`, response.status);
                return response.json();
            })
            .then(data => {
                console.log(`[uploadSourceAndIndex] Data received:`, data);
                if (data.success) {
                    // Clear input fields
                    const textInput = document.getElementById(`source-text-input-${projectId}`);
                    const fileInput = document.getElementById(`source-file-input-${projectId}`);
                    if (textInput) textInput.value = '';
                    if (fileInput) fileInput.value = '';
                    
                    // Start polling for indexing progress
                    const trackingIds = data.tracking_ids || [];
                    console.log(`[uploadSourceAndIndex] Tracking IDs:`, trackingIds);
                    if (trackingIds.length > 0) {
                        pollIndexingProgress(projectId, trackingIds, indexingContainer, statusMessage, progressFill, progressText, indexingStatus, indexingErrors);
                    }
                } else {
                    console.error(`[uploadSourceAndIndex] Error:`, data.error);
                    showError(statusMessage, data.error);
                    indexingContainer.style.display = 'none';
                }
            })
            .catch(error => {
                console.error('[uploadSourceAndIndex] Fetch error:', error);
                showError(statusMessage, 'Failed to add source: ' + error.message);
                indexingContainer.style.display = 'none';
            });
    } catch (err) {
        console.error('[uploadSourceAndIndex] Exception:', err);
        alert('Error: ' + err.message);
    }
}

function pollIndexingProgress(projectId, trackingIds, indexingContainer, statusMessage, progressFill, progressText, indexingStatus, indexingErrors) {
    let completedCount = 0;
    let errorCount = 0;
    const maxPolls = 300; // 5 minutes with 1-second intervals
    let pollCount = 0;
    
    const pollInterval = setInterval(() => {
        pollCount++;
        
        // Check all tracking IDs
        Promise.all(trackingIds.map(trackingId => 
            fetch(buildUrl(`api/indexing/progress/${trackingId}`))
                .then(r => r.json())
                .catch(() => ({ status: 'unknown' }))
        ))
        .then(results => {
            completedCount = 0;
            errorCount = 0;
            let avgProgress = 0;
            let statuses = [];
            let allErrors = [];
            
            results.forEach(result => {
                if (result.status === 'completed') {
                    completedCount++;
                }
                if (result.status === 'completed_with_errors' || result.status === 'failed') {
                    errorCount++;
                }
                statuses.push(result.status);
                avgProgress += result.progress || 0;
                if (result.errors && result.errors.length > 0) {
                    allErrors.push(...result.errors);
                }
            });
            
            avgProgress = Math.round(avgProgress / results.length);
            updateProgress(progressFill, progressText, avgProgress);
            
            // Update status message
            if (completedCount === trackingIds.length) {
                // All completed
                indexingStatus.textContent = '✅ Indexing completed!';
                
                if (errorCount > 0) {
                    showWarning(statusMessage, `Indexing completed with ${errorCount} error(s)`);
                    displayErrors(indexingErrors, allErrors);
                } else {
                    showSuccess(statusMessage, 'All sources indexed successfully');
                }
                
                clearInterval(pollInterval);
                
                // Refresh project details after a delay
                setTimeout(() => selectProject(projectId), 2000);
            } else {
                indexingStatus.textContent = `Processing: ${completedCount}/${trackingIds.length} sources completed`;
            }
        })
        .catch(error => {
            console.error('Error polling progress:', error);
        });
        
        // Timeout after maxPolls
        if (pollCount >= maxPolls) {
            clearInterval(pollInterval);
            showWarning(statusMessage, 'Indexing is taking longer than expected. Please check again later.');
            indexingContainer.style.display = 'none';
        }
    }, 1000);
}

function updateProgress(progressFill, progressText, percent) {
    const rounded = Math.round(percent);
    progressFill.style.width = rounded + '%';
    progressText.textContent = rounded + '%';
}

function displayErrors(errorContainer, errors) {
    if (!errors || errors.length === 0) return;
    
    let html = '<strong>⚠️ Errors encountered:</strong>';
    errors.forEach(err => {
        html += `<div class="error-item"><span class="error-type">${err.type}:</span> ${err.message}</div>`;
    });
    
    errorContainer.innerHTML = html;
    errorContainer.style.display = 'block';
}

function showSuccess(element, message) {
    element.textContent = '✅ ' + message;
    element.className = 'status-message success';
    element.style.display = 'block';
    
    setTimeout(() => {
        element.style.display = 'none';
    }, 7000);
}

function showError(element, message) {
    element.textContent = '❌ ' + message;
    element.className = 'status-message error';
    element.style.display = 'block';
}

function showWarning(element, message) {
    element.textContent = '⚠️ ' + message;
    element.className = 'status-message warning';
    element.style.display = 'block';
}

function deleteProject(projectId, projectName) {
    if (confirm(`Are you sure you want to delete "${projectName}"? This will also delete all its sources.`)) {
        fetch(buildUrl(`api/projects/${projectId}/delete`), { method: 'DELETE' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Reload projects list
                    const container = document.getElementById('projects-container');
                    if (container) {
                        fetch(buildUrl('api/projects'))
                            .then(response => response.text())
                            .then(html => {
                                container.innerHTML = html;
                            })
                            .catch(error => {
                                console.error('Error loading projects:', error);
                                container.innerHTML = '<div class="error">Failed to load projects</div>';
                            });
                    }
                    // Clear project details
                    const detailsContainer = document.getElementById('project-details-container');
                    if (detailsContainer) {
                        detailsContainer.innerHTML = '<div class="card"><p class="text-muted">Select a project from the list to manage its sources</p></div>';
                    }
                } else {
                    alert('Error deleting project: ' + data.error);
                }
            })
            .catch(error => console.error('Error:', error));
    }
}

function removeSource(projectId, source) {
    if (confirm(`Remove source: ${source}?`)) {
        const formData = new FormData();
        formData.append('project_id', projectId);
        formData.append('source', source);
        
        fetch('api/projects/sources/remove', { method: 'POST', body: formData })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    selectProject(projectId);
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => console.error('Error:', error));
    }
}

function selectProject(projectId) {
    fetch(buildUrl(`api/projects/${projectId}/details`))
        .then(response => response.text())
        .then(html => {
            document.getElementById('project-details-container').innerHTML = html;
        })
        .catch(error => {
            console.error('Error loading project details:', error);
        });
}

/**
 * Indexing Progress Polling Script
 * Handles real-time progress updates for source indexing
 */

/**
 * Poll the indexing progress endpoint and update the UI
 * @param {string} projectId - The project ID
 * @param {array} trackingIds - Array of tracking IDs to monitor
 * @param {HTMLElement} container - Progress container element
 * @param {HTMLElement} statusMessage - Status message element
 * @param {HTMLElement} progressFill - Progress bar fill element
 * @param {HTMLElement} progressText - Progress percentage text element
 * @param {HTMLElement} indexingStatus - Indexing status text element
 * @param {HTMLElement} indexingErrors - Error container element
 */
function pollIndexingProgress(projectId, trackingIds, container, statusMessage, progressFill, progressText, indexingStatus, indexingErrors) {
    console.log(`[pollIndexingProgress] Starting poll for ${trackingIds.length} tracking IDs`);
    let completedCount = 0;
    let errorCount = 0;
    const maxPolls = 300; // 5 minutes with 1-second intervals
    let pollCount = 0;
    
    const pollInterval = setInterval(() => {
        pollCount++;
        
        // Fetch progress for all tracking IDs in parallel
        Promise.all(trackingIds.map(trackingId => 
            fetch(`/api/indexing/progress/${trackingId}`)
                .then(r => r.json())
                .catch(err => {
                    console.error(`Failed to get progress for ${trackingId}:`, err);
                    return { status: 'unknown', progress: 0, errors: [] };
                })
        ))
        .then(results => {
            completedCount = 0;
            errorCount = 0;
            let avgProgress = 0;
            let allErrors = [];
            
            results.forEach(result => {
                // Count completed items
                if (result.status === 'completed') {
                    completedCount++;
                }
                
                // Count items with errors
                if (result.status === 'error') {
                    errorCount++;
                }
                
                // Accumulate progress
                avgProgress += result.progress || 0;
                
                // Collect errors
                if (result.errors && Array.isArray(result.errors)) {
                    allErrors.push(...result.errors);
                }
            });
            
            // Calculate average progress
            avgProgress = Math.round(avgProgress / results.length);
            updateProgressBar(progressFill, progressText, avgProgress);
            
            // Update status message
            if (completedCount === trackingIds.length) {
                // All completed (some may have errors)
                indexingStatus.textContent = '✅ Indexing completed!';
                
                if (errorCount > 0) {
                    showWarning(statusMessage, `Indexing completed with ${errorCount} error(s)`);
                    displayErrors(indexingErrors, allErrors);
                } else {
                    showSuccess(statusMessage, 'All sources indexed successfully');
                }
                
                clearInterval(pollInterval);
                
                // Refresh project details after a short delay
                setTimeout(() => selectProject(projectId), 2000);
            } else {
                // Still processing
                indexingStatus.textContent = `Processing: ${completedCount}/${trackingIds.length} sources completed`;
                
                if (allErrors.length > 0) {
                    displayErrors(indexingErrors, allErrors);
                }
            }
        })
        .catch(error => {
            console.error('Error polling progress:', error);
            showError(statusMessage, 'Error polling indexing progress: ' + error.message);
        });
        
        // Stop polling after max attempts
        if (pollCount >= maxPolls) {
            clearInterval(pollInterval);
            showWarning(statusMessage, 'Indexing is taking longer than expected. Please check again later.');
            container.style.display = 'none';
        }
    }, 1000); // Poll every 1 second
}

/**
 * Update the progress bar with current progress
 * @param {HTMLElement} progressFill - Progress bar fill element
 * @param {HTMLElement} progressText - Progress text element
 * @param {number} percent - Progress percentage (0-100)
 */
function updateProgressBar(progressFill, progressText, percent) {
    const rounded = Math.round(Math.min(Math.max(percent, 0), 100));
    progressFill.style.width = rounded + '%';
    progressText.textContent = rounded + '%';
}

/**
 * Display errors in the UI
 * @param {HTMLElement} errorContainer - Error container element
 * @param {array} errors - Array of error strings or objects
 */
function displayErrors(errorContainer, errors) {
    if (!errors || errors.length === 0) return;
    
    let html = '<strong>⚠️ Errors encountered:</strong>';
    
    errors.forEach(err => {
        if (typeof err === 'object' && err.message) {
            html += `<div class="error-item"><span class="error-type">${err.type || 'Error'}:</span> ${err.message}</div>`;
        } else {
            html += `<div class="error-item">${err}</div>`;
        }
    });
    
    errorContainer.innerHTML = html;
    errorContainer.style.display = 'block';
}

/**
 * Show a success message
 * @param {HTMLElement} element - Message element
 * @param {string} message - Message text
 */
function showSuccess(element, message) {
    element.textContent = '✅ ' + message;
    element.className = 'status-message success';
    element.style.display = 'block';
    
    // Auto-hide after 7 seconds
    setTimeout(() => {
        element.style.display = 'none';
    }, 7000);
}

/**
 * Show an error message
 * @param {HTMLElement} element - Message element
 * @param {string} message - Message text
 */
function showError(element, message) {
    element.textContent = '❌ ' + message;
    element.className = 'status-message error';
    element.style.display = 'block';
}

/**
 * Show a warning message
 * @param {HTMLElement} element - Message element
 * @param {string} message - Message text
 */
function showWarning(element, message) {
    element.textContent = '⚠️ ' + message;
    element.className = 'status-message warning';
    element.style.display = 'block';
}

/**
 * Show an info message
 * @param {HTMLElement} element - Message element
 * @param {string} message - Message text
 */
function showInfo(element, message) {
    element.textContent = 'ℹ️ ' + message;
    element.className = 'status-message info';
    element.style.display = 'block';
}

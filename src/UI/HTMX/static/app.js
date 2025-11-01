/**
 * HTMX SPA Application Script
 * Handles page transitions and active state management
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the app
    initializeApp();
});

function initializeApp() {
    // Listen for HTMX events
    document.body.addEventListener('htmx:afterRequest', handlePageTransition);
    document.body.addEventListener('htmx:configRequest', updateActiveLink);
}

/**
 * Update the active navigation link based on the current URL
 */
function updateActiveLink(event) {
    const links = document.querySelectorAll('.nav-link');
    links.forEach(link => link.classList.remove('active'));
    
    // Find the link that matches the current href
    const currentHref = window.location.pathname;
    const activeLink = document.querySelector(`.nav-link[href="${currentHref}"]`);
    if (activeLink) {
        activeLink.classList.add('active');
    }
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

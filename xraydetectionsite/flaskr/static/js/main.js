// Enhanced X-Ray Detection Site JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const fileInput = document.querySelector('input[type="file"]');
    const form = document.querySelector('form');
    const submitButton = document.querySelector('button[type="submit"]');
    const uploadSection = document.getElementById('upload');
    
    // File validation settings
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
    const maxFileSize = 10 * 1024 * 1024; // 10MB
    
    // Initialize the application
    init();
    
    function init() {
        setupFileInput();
        setupFormSubmission();
        setupAccessibility();
        addLoadingStates();
        createFilePreview();
    }
    
    function setupFileInput() {
        if (!fileInput) return;
        
        // Get the file drop zone
        const fileDropZone = document.querySelector('.file-drop-zone');
        if (!fileDropZone) return;
        
        // Add drag and drop functionality
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileDropZone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        
        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            fileDropZone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            fileDropZone.addEventListener(eventName, unhighlight, false);
        });
        
        // Handle dropped files
        fileDropZone.addEventListener('drop', handleDrop, false);
        
        // Handle file selection
        fileInput.addEventListener('change', handleFileSelect, false);
        
        // Handle click on drop zone
        fileDropZone.addEventListener('click', function(e) {
            if (e.target !== fileInput) {
                fileInput.click();
            }
        });
    }
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight(e) {
        const container = e.currentTarget;
        container.style.borderColor = '#764ba2';
        container.style.backgroundColor = 'rgba(118, 75, 162, 0.1)';
        container.style.transform = 'scale(1.02)';
    }
    
    function unhighlight(e) {
        const container = e.currentTarget;
        container.style.borderColor = '#667eea';
        container.style.backgroundColor = 'rgba(102, 126, 234, 0.05)';
        container.style.transform = 'scale(1)';
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect({ target: { files: files } });
        }
    }
    
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        // Validate file
        if (!validateFile(file)) {
            return;
        }
        
        // Show file preview
        showFilePreview(file);
        
        // Update submit button state
        updateSubmitButton(true);
        
        // Add success animation
        animateSuccess();
    }
    
    function validateFile(file) {
        // Check file type
        if (!allowedTypes.includes(file.type)) {
            showNotification('Please select a valid image file (JPEG, PNG, GIF, or WebP)', 'error');
            return false;
        }
        
        // Check file size
        if (file.size > maxFileSize) {
            showNotification('File size must be less than 10MB', 'error');
            return false;
        }
        
        return true;
    }
    
    function showFilePreview(file) {
        // Remove existing preview
        const existingPreview = document.getElementById('file-preview');
        if (existingPreview) {
            existingPreview.remove();
        }
        
        // Create preview container
        const previewContainer = document.createElement('div');
        previewContainer.id = 'file-preview';
        previewContainer.style.cssText = `
            margin-top: 1.5rem;
            padding: 1rem;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 10px;
            border: 1px solid rgba(102, 126, 234, 0.3);
            animation: fadeIn 0.3s ease;
        `;
        
        // Create file info
        const fileInfo = document.createElement('div');
        fileInfo.style.cssText = `
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        `;
        
        const fileName = document.createElement('span');
        fileName.textContent = file.name;
        fileName.style.cssText = `
            font-weight: 600;
            color: #2c3e50;
        `;
        
        const fileSize = document.createElement('span');
        fileSize.textContent = formatFileSize(file.size);
        fileSize.style.cssText = `
            color: #666;
            font-size: 0.9rem;
        `;
        
        fileInfo.appendChild(fileName);
        fileInfo.appendChild(fileSize);
        
        // Create image preview
        const img = document.createElement('img');
        img.style.cssText = `
            max-width: 100%;
            max-height: 200px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        `;
        
        // Read file and create preview
        const reader = new FileReader();
        reader.onload = function(e) {
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        previewContainer.appendChild(fileInfo);
        previewContainer.appendChild(img);
        
        // Insert preview after form
        form.parentNode.insertBefore(previewContainer, form.nextSibling);
    }
    
    function setupFormSubmission() {
        if (!form) return;
        
        form.addEventListener('submit', function(e) {
            // Add loading state
            submitButton.classList.add('loading');
            submitButton.disabled = true;
            submitButton.textContent = 'Uploading...';
            
            // You can add additional validation here
            // For now, we'll let the form submit normally
        });
    }
    
    function setupAccessibility() {
        // Add ARIA labels and descriptions
        if (fileInput) {
            fileInput.setAttribute('aria-label', 'Select X-ray image file');
            fileInput.setAttribute('aria-describedby', 'file-requirements');
            
            // Create requirements description
            const requirements = document.createElement('div');
            requirements.id = 'file-requirements';
            requirements.style.cssText = `
                font-size: 0.9rem;
                color: #666;
                margin-top: 0.5rem;
                text-align: center;
            `;
            requirements.textContent = 'Accepted formats: JPEG, PNG, GIF, WebP. Max size: 10MB';
            
            fileInput.parentNode.appendChild(requirements);
        }
        
        // Add keyboard navigation for custom elements
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                if (e.target.classList.contains('file-drop-zone')) {
                    e.target.querySelector('input[type="file"]').click();
                }
            }
        });
    }
    
    function addLoadingStates() {
        // Add CSS for loading animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 1rem 1.5rem;
                border-radius: 8px;
                color: white;
                font-weight: 500;
                z-index: 1000;
                animation: slideIn 0.3s ease;
                max-width: 300px;
            }
            
            .notification.success {
                background: linear-gradient(135deg, #4CAF50, #45a049);
            }
            
            .notification.error {
                background: linear-gradient(135deg, #f44336, #da190b);
            }
            
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
    
    function createFilePreview() {
        // This function is called during initialization
        // File preview creation is handled in showFilePreview function
    }
    
    function updateSubmitButton(enabled) {
        if (!submitButton) return;
        
        submitButton.disabled = !enabled;
        if (enabled) {
            submitButton.textContent = 'Analyze X-Ray';
            submitButton.style.opacity = '1';
        } else {
            submitButton.textContent = 'Select a file first';
            submitButton.style.opacity = '0.6';
        }
    }
    
    function animateSuccess() {
        // Add success animation to the upload section
        uploadSection.style.animation = 'none';
        uploadSection.offsetHeight; // Trigger reflow
        uploadSection.style.animation = 'fadeIn 0.5s ease';
    }
    
    function showNotification(message, type = 'info') {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(notification => notification.remove());
        
        // Create new notification
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Auto-remove notification after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Initialize submit button state
    updateSubmitButton(false);
    
    // Add smooth scrolling for any anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add intersection observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animation = 'fadeIn 0.6s ease forwards';
            }
        });
    }, observerOptions);
    
    // Observe all sections
    document.querySelectorAll('section').forEach(section => {
        observer.observe(section);
    });
});
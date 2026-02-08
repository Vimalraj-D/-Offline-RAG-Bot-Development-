// Upload Modal Management
const uploadModal = document.getElementById('uploadModal');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const selectedFilesDiv = document.getElementById('selectedFiles');
const uploadButton = document.getElementById('uploadButton');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const documentsList = document.getElementById('documentsList');

let selectedFiles = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupUploadListeners();
    loadDocuments();
});

// Setup event listeners for upload
function setupUploadListeners() {
    // Click on upload area
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files);
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        handleFileSelect(e.dataTransfer.files);
    });
}

// Open upload modal
function openUploadModal() {
    uploadModal.classList.add('active');
    selectedFiles = [];
    updateSelectedFiles();
}

// Close upload modal
function closeUploadModal() {
    uploadModal.classList.remove('active');
    selectedFiles = [];
    fileInput.value = '';
    updateSelectedFiles();
    uploadProgress.classList.remove('active');
    progressFill.style.width = '0%';
}

// Handle file selection
function handleFileSelect(files) {
    const newFiles = Array.from(files);
    
    newFiles.forEach(file => {
        // Check file size (16MB limit)
        if (file.size > 16 * 1024 * 1024) {
            alert(`File ${file.name} is too large. Maximum size is 16MB.`);
            return;
        }
        
        const ext = file.name.split('.').pop().toLowerCase();
        if (!['pdf', 'txt', 'md', 'docx', 'csv'].includes(ext)) {
            alert(`File ${file.name} has an unsupported format. Use PDF, TXT, MD, DOCX, or CSV.`);
            return;
        }
        
        // Check if file already selected
        if (!selectedFiles.find(f => f.name === file.name)) {
            selectedFiles.push(file);
        }
    });
    
    updateSelectedFiles();
}

// Update selected files display
function updateSelectedFiles() {
    if (selectedFiles.length === 0) {
        selectedFilesDiv.innerHTML = '';
        uploadButton.disabled = true;
        return;
    }
    
    selectedFilesDiv.innerHTML = selectedFiles.map((file, index) => `
        <div class="file-item">
            <div class="file-info">
                <span class="file-icon">${getFileIcon(file.name)}</span>
                <div>
                    <div class="file-name">${file.name}</div>
                    <span class="file-size">${formatFileSize(file.size)}</span>
                </div>
            </div>
            <button class="remove-file" onclick="removeFile(${index})">
                ✕
            </button>
        </div>
    `).join('');
    
    uploadButton.disabled = false;
}

// Remove file from selection
function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateSelectedFiles();
}

// Get file icon based on extension
function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const icons = { 'pdf': '📕', 'txt': '📄', 'md': '📝', 'docx': '📘', 'csv': '📊' };
    return icons[ext] || '📄';
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Upload files
async function uploadFiles() {
    if (selectedFiles.length === 0) return;
    
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });
    
    // Disable button and show progress
    uploadButton.disabled = true;
    uploadButton.textContent = 'Uploading...';
    uploadProgress.classList.add('active');
    progressFill.style.width = '10%';
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        progressFill.style.width = '50%';
        
        const data = await response.json();
        
        progressFill.style.width = '100%';
        
        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }
        
        // Show success message
        alert(`Successfully uploaded and processed ${data.uploaded.length} document(s)!\n\nTotal chunks: ${data.total_chunks}`);
        
        // Close modal and refresh
        setTimeout(() => {
            closeUploadModal();
            loadDocuments();
            loadStats();
        }, 500);
        
    } catch (error) {
        console.error('Upload error:', error);
        alert(`Error uploading files: ${error.message}`);
        uploadProgress.classList.remove('active');
        uploadButton.disabled = false;
        uploadButton.textContent = 'Upload & Process';
    }
}

// Load documents list
async function loadDocuments() {
    try {
        const response = await fetch('/api/documents');
        const data = await response.json();
        
        if (data.documents && data.documents.length > 0) {
            documentsList.innerHTML = data.documents.map(doc => `
                <div class="document-item">
                    <div class="doc-info">
                        <div class="doc-name">${getFileIcon(doc.name)} ${doc.name}</div>
                        <div class="doc-size">${formatFileSize(doc.size)}</div>
                    </div>
                    <button class="delete-doc-button" onclick="deleteDocument('${doc.name}')">
                        🗑️
                    </button>
                </div>
            `).join('');
        } else {
            documentsList.innerHTML = '<p class="empty-state">No documents yet. Upload some to get started!</p>';
        }
    } catch (error) {
        console.error('Error loading documents:', error);
        documentsList.innerHTML = '<p class="empty-state">Error loading documents</p>';
    }
}

// Delete document
async function deleteDocument(filename) {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) {
        return;
    }
    
    try {
        showLoading(true);
        
        const response = await fetch(`/api/documents/${filename}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Delete failed');
        }
        
        // Refresh documents list and stats
        await loadDocuments();
        await loadStats();
        
        alert(`Document deleted successfully!`);
        
    } catch (error) {
        console.error('Delete error:', error);
        alert(`Error deleting document: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Make functions globally available
window.openUploadModal = openUploadModal;
window.closeUploadModal = closeUploadModal;
window.removeFile = removeFile;
window.uploadFiles = uploadFiles;
window.deleteDocument = deleteDocument;

// Refresh documents list periodically
setInterval(loadDocuments, 30000); // Every 30 seconds
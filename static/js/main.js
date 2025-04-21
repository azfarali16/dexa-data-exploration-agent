const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const uploadBtn = document.querySelector('.upload-btn');
const uploadTxt = document.getElementById('uploadtxt');

let isFileUploaded = false;
let uploadedFile = null;

// Handle drag over event to indicate valid drop area
uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.style.borderColor = '#4CAF50';
});

// Handle drag leave event to reset the drop area border color
uploadArea.addEventListener('dragleave', () => {
  uploadArea.style.borderColor = '#ccc';
});

// Handle file drop event
uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  handleFileUpload(e.dataTransfer.files[0]);
});

// Handle file input change (file selected from file dialog)
fileInput.addEventListener('change', () => {
  handleFileUpload(fileInput.files[0]);
});

// Function to handle file upload
function handleFileUpload(file) {

  const validExtensions = ['.csv', '.xlsx', '.xls'];
  const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();

  if (!validExtensions.includes(fileExtension)) {
    isFileUploaded = true;
    uploadedFile = file;
    uploadArea.style.borderColor = '#4CAF50';
    console.log(file.name);
    console.log(uploadTxt)
    console.log(uploadTxt.textContent);
    uploadTxt.textContent = file.name;
    analyzeBtn.classList.add('active'); 
  } else {
    alert('Please upload a valid CSV or XLSX file');
  }
}

// Handle analyze button click event
analyzeBtn.addEventListener('click', () => {
  if (isFileUploaded) {
    const formData = new FormData();
    formData.append("file", uploadedFile);

    // Send file to Flask backend using fetch (AJAX)
    fetch("/analyze", {
      method: "POST",
      body: formData
    })
    .then(response => {
      if (response.ok) {
        // After success, redirect to analytics page while keeping the button visible
        window.location.href = "/analytics"; // Redirect to analytics page
      } else {
        alert("Error: Unable to process the file");
      }
    })
    .catch(error => {
      console.error('Error during file upload:', error);
      alert("Error: Unable to process the file");
    });
  } else {
    alert('No file uploaded yet!');
  }
});

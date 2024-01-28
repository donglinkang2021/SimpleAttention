// Function to fetch data from result.json
function fetchData() {
    fetch('result.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            populateDropdowns(data);
            updateDropdowns();
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
}

// Function to populate dropdowns with data
function populateDropdowns(data) {
    const graphDropdown = document.getElementById('graphName');
    const modelDropdown = document.getElementById('modelName');
    const datasetDropdown = document.getElementById('datasetName');

    clearDropdown(graphDropdown);
    clearDropdown(modelDropdown);
    clearDropdown(datasetDropdown);

    data.graphNames.forEach(option => addOptionToDropdown(graphDropdown, option));
    data.modelNames.forEach(option => addOptionToDropdown(modelDropdown, option));
    data.datasetNames.forEach(option => addOptionToDropdown(datasetDropdown, option));
}

// Function to clear dropdown options
function clearDropdown(dropdown) {
    dropdown.innerHTML = '';
}

// Function to add option to dropdown
function addOptionToDropdown(dropdown, option) {
    const newOption = document.createElement('option');
    newOption.value = option;
    newOption.text = option;
    dropdown.add(newOption);
}

// Event listener for window onload
window.onload = function () {
    fetchData();
};

// Event listener for dropdown changes
document.getElementById('graphName').addEventListener('change', updateDropdowns);
document.getElementById('modelName').addEventListener('change', updateDropdowns);
document.getElementById('datasetName').addEventListener('change', updateDropdowns);

// Function to update dropdowns based on selected options
function updateDropdowns() {
    const selectedGraph = document.getElementById('graphName').value;
    const selectedModel = document.getElementById('modelName').value;
    const selectedDataset = document.getElementById('datasetName').value;

    const imagePath = get_png_name(selectedGraph, selectedModel, selectedDataset);

    const imageContainer = document.getElementById('imageContainer');
    imageContainer.innerHTML = `<img src="${imagePath}" alt="Generated Image">`;
}

function get_png_name(selectedGraph, selectedModel, selectedDataset) {
    return `../result/${selectedGraph}_for_model_${selectedModel}_dataset_${selectedDataset}.png`;
}

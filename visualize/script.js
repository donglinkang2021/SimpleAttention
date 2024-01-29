var json_data;

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
            json_data = data;
            populateDropdowns();
            updateDropdowns();
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
}

// Function to populate dropdowns with data
function populateDropdowns(){
    const graphDropdown = document.getElementById('graphName');
    const modelDropdown = document.getElementById('modelName');
    const datasetDropdown = document.getElementById('datasetName');

    clearDropdown(graphDropdown);
    clearDropdown(modelDropdown);
    clearDropdown(datasetDropdown);

    json_data.datasetNames.forEach(option => addOptionToDropdown(datasetDropdown, option));
    json_data.modelNames.forEach(option => addOptionToDropdown(modelDropdown, option));
    json_data.graphNames.forEach(option => addOptionToDropdown(graphDropdown, option));

    updateDataset();
}

function updateDataset(){
    const selectedDataset = document.getElementById('datasetName').value;
    const modelDropdown = document.getElementById('modelName');

    clearDropdown(modelDropdown);

    const dataset_model = json_data.model[`${selectedDataset}`];
    dataset_model.forEach(option => addOptionToDropdown(modelDropdown, option));

    updateDropdowns();
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


window.onload = function() {
    fetchData();
};


// Function to update dropdowns based on selected options
function updateDropdowns() {
    console.log("Dropdowns updated!");

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

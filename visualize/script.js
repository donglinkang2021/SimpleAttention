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
    const datasetDropdown = document.getElementById('datasetName');
    const modelDropdown = document.getElementById('modelName');
    
    clearDropdown(datasetDropdown);
    clearDropdown(modelDropdown);

    json_data.datasetNames.forEach(option => addOptionToDropdown(datasetDropdown, option));
    json_data.modelNames.forEach(option => addOptionToDropdown(modelDropdown, option));

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
    const selectedDataset = document.getElementById('datasetName').value;
    const selectedModel = document.getElementById('modelName').value;    
    const graphDB = json_data.graphNames[0];
    const graphLoss = json_data.graphNames[1];
    const imagePathDB = get_png_name(graphDB, selectedModel, selectedDataset);
    const imagePathLoss = get_png_name(graphLoss, selectedModel, selectedDataset);
    const imageContainerDB = document.getElementById('imageContainerDB');
    const imageContainerLoss = document.getElementById('imageContainerLoss');
    imageContainerDB.innerHTML = `<img src="${imagePathDB}" alt="Decision Boundary" width="100%">`;
    imageContainerLoss.innerHTML = `<img src="${imagePathLoss}" alt="Loss" width="100%">`;
}

function get_png_name(selectedGraph, selectedModel, selectedDataset) {
    return `../result/${selectedGraph}_for_model_${selectedModel}_dataset_${selectedDataset}.png`;
}

const graphOptions = ['MSE_per_batch', 'Decision_boundary'];

const modelOptions = ['heads2_embed64_no_act', 'heads2_embed8_no_act', 'heads8_embed32_no_act', 'linear_hid2_embed16_tanh', 'linear_hid1_embed16_relu', 'heads8_embed16_no_act', 'heads16_embed64_no_act', 'heads8_embed64_no_act', 'heads2_embed4_no_act', 'heads16_embed16_no_act', 'heads4_embed64_no_act', 'heads8_embed8_no_act', 'heads4_embed16_no_act', 'heads4_embed8_no_act', 'linear_hid2_embed8_tanh', 'heads16_embed32_no_act', 'heads4_embed32_no_act', 'heads2_embed32_no_act', 'heads32_embed64_no_act', 'linear_hid2_embed4_tanh', 'heads64_embed64_no_act', 'heads4_embed4_no_act', 'linear_hid1_embed32_relu', 'linear_hid1_embed8_relu', 'linear_hid1_embed4_relu', 'linear_hid2_embed32_tanh', 'heads2_embed16_no_act', 'heads32_embed32_no_act'];

const datasetOptions = ['regress_plane', 'regress_gaussian'];


window.onload = function() {
    populateDropdowns();
    updateDropdowns();
}

// Function to populate dropdowns with data
function populateDropdowns(){
    const graphDropdown = document.getElementById('graphName');
    const modelDropdown = document.getElementById('modelName');
    const datasetDropdown = document.getElementById('datasetName');

    clearDropdown(graphDropdown);
    clearDropdown(modelDropdown);
    clearDropdown(datasetDropdown);

    graphOptions.forEach(option => addOptionToDropdown(graphDropdown, option));
    modelOptions.forEach(option => addOptionToDropdown(modelDropdown, option));
    datasetOptions.forEach(option => addOptionToDropdown(datasetDropdown, option));
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

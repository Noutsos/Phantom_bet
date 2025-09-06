document.addEventListener('DOMContentLoaded', function() {
    // Form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            let valid = true;
            const requiredFields = form.querySelectorAll('[required]');
            
            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    valid = false;
                    field.style.borderColor = 'red';
                } else {
                    field.style.borderColor = '';
                }
            });
            
            if (!valid) {
                e.preventDefault();
                alert('Please fill in all required fields.');
            }
        });
    });
    
    
    // Dynamic form behavior
    const collectionPhase = document.getElementById('collection_phase');
    if (collectionPhase) {
        collectionPhase.addEventListener('change', function() {
            // You can add dynamic behavior based on collection phase selection
            console.log('Collection phase changed to:', this.value);
        });
    }
    
    // Toggle advanced options
    const advancedOptions = document.querySelectorAll('.advanced-option');
    advancedOptions.forEach(option => {
        const toggle = option.querySelector('.advanced-toggle');
        const content = option.querySelector('.advanced-content');
        
        if (toggle && content) {
            toggle.addEventListener('click', function() {
                content.style.display = content.style.display === 'none' ? 'block' : 'none';
                toggle.textContent = content.style.display === 'none' ? 'Show Advanced Options' : 'Hide Advanced Options';
            });
            
            // Hide by default
            content.style.display = 'none';
        }
    });
});

// Add search functionality
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = 'Search leagues...';
    searchInput.className = 'form-control mb-2';
    searchInput.oninput = function(e) {
        const searchTerm = e.target.value.toLowerCase();
        document.querySelectorAll('.checkbox-label').forEach(label => {
            const text = label.textContent.toLowerCase();
            label.style.display = text.includes(searchTerm) ? 'block' : 'none';
        });
    };
    
    const leagueSelection = document.querySelector('.league-selection');
    leagueSelection.parentNode.insertBefore(searchInput, leagueSelection);
});

function selectAll() {
    document.querySelectorAll('input[name="selected_leagues"]').forEach(checkbox => {
        checkbox.checked = true;
    });
}

function deselectAll() {
    document.querySelectorAll('input[name="selected_leagues"]').forEach(checkbox => {
        checkbox.checked = false;
    });
}

function selectTopTier() {
    document.querySelectorAll('input[name="selected_leagues"]').forEach(checkbox => {
        const label = checkbox.parentElement.textContent;
        checkbox.checked = label.includes('top_tier') || label.includes('1st');
    });
}

function selectDomesticCups() {
    document.querySelectorAll('input[name="selected_leagues"]').forEach(checkbox => {
        const label = checkbox.parentElement.textContent;
        checkbox.checked = label.includes('domestic_cup') || label.includes('cup');
    });
}

// Toggle form fields based on selections
function toggleImbalanceFields() {
    const handleImbalance = document.getElementById('handle_class_imbalance').checked;
    const classWeightType = document.getElementById('class_weight_type').value;
    const useSmote = document.getElementById('use_smote').checked;
    
    // Show/hide class weight options
    document.getElementById('class_weight_type').style.display = handleImbalance && !useSmote ? 'block' : 'none';
    document.getElementById('custom_weights_group').style.display = (handleImbalance && !useSmote && classWeightType === 'custom') ? 'block' : 'none';
    
    // Show/hide SMOTE options
    document.getElementById('smote_strategy_group').style.display = (handleImbalance && useSmote) ? 'block' : 'none';
    document.getElementById('smote_factor_group').style.display = (handleImbalance && useSmote) ? 'block' : 'none';
}


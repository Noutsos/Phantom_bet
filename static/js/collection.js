// Toggle node expansion
function toggleNode(btn) {
    const children = btn.closest('.node-header').nextElementSibling;
    if (children.style.display === 'none') {
        children.style.display = 'block';
        btn.textContent = '▼';
    } else {
        children.style.display = 'none';
        btn.textContent = '▶';
    }
}

// Toggle entire region
function toggleRegion(region, checked) {
    const checkboxes = document.querySelectorAll(`
        .region-node[data-region="${region}"] .country-checkbox,
        .region-node[data-region="${region}"] .league-checkbox
    `);
    checkboxes.forEach(cb => cb.checked = checked);
    updateRegionState(region);
}

// Toggle entire country
function toggleCountry(region, country, checked) {
    const checkboxes = document.querySelectorAll(`
        .country-node[data-region="${region}"][data-country="${country}"] .league-checkbox
    `);
    checkboxes.forEach(cb => cb.checked = checked);
    updateCountryState(region, country);
    updateRegionState(region);
}

// Update parent states when leagues are selected
function updateParentStates(region, country) {
    updateCountryState(region, country);
    updateRegionState(region);
}

function updateCountryState(region, country) {
    const countryCheckbox = document.querySelector(`
        .country-node[data-region="${region}"][data-country="${country}"] .country-checkbox
    `);
    const leagueCheckboxes = document.querySelectorAll(`
        .country-node[data-region="${region}"][data-country="${country}"] .league-checkbox
    `);
    
    const allChecked = Array.from(leagueCheckboxes).every(cb => cb.checked);
    const someChecked = Array.from(leagueCheckboxes).some(cb => cb.checked);
    
    countryCheckbox.checked = allChecked;
    countryCheckbox.indeterminate = someChecked && !allChecked;
}

function updateRegionState(region) {
    const regionCheckbox = document.querySelector(`
        .region-node[data-region="${region}"] .region-checkbox
    `);
    const countryCheckboxes = document.querySelectorAll(`
        .region-node[data-region="${region}"] .country-checkbox
    `);
    
    const allCountriesChecked = Array.from(countryCheckboxes).every(cb => cb.checked && !cb.indeterminate);
    const someChecked = Array.from(countryCheckboxes).some(cb => cb.checked || cb.indeterminate);
    
    regionCheckbox.checked = allCountriesChecked;
    regionCheckbox.indeterminate = someChecked && !allCountriesChecked;
}

// Quick selection functions
function selectAll() {
    document.querySelectorAll('.region-checkbox, .country-checkbox, .league-checkbox').forEach(cb => {
        cb.checked = true;
    });
    document.querySelectorAll('.region-node').forEach(node => {
        const region = node.getAttribute('data-region');
        updateRegionState(region);
    });
}

function deselectAll() {
    document.querySelectorAll('.region-checkbox, .country-checkbox, .league-checkbox').forEach(cb => {
        cb.checked = false;
        cb.indeterminate = false;
    });
}

function selectByCategory(category) {
    document.querySelectorAll('.league-checkbox').forEach(cb => {
        const leagueCategory = cb.closest('.league-node').getAttribute('data-category');
        cb.checked = (leagueCategory === category);
    });
    // Update all parent states
    document.querySelectorAll('.region-node').forEach(node => {
        const region = node.getAttribute('data-region');
        const countries = node.querySelectorAll('.country-node');
        countries.forEach(countryNode => {
            const country = countryNode.getAttribute('data-country');
            updateCountryState(region, country);
        });
        updateRegionState(region);
    });
}

// Search functionality
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('unifiedSearch');
    if (searchInput) {
        searchInput.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            
            document.querySelectorAll('.region-node, .country-node, .league-node').forEach(node => {
                const text = node.textContent.toLowerCase();
                if (text.includes(searchTerm)) {
                    node.style.display = 'block';
                    // Expand parent nodes
                    let parent = node.parentElement;
                    while (parent && parent.classList.contains('node-children')) {
                        parent.style.display = 'block';
                        const header = parent.previousElementSibling;
                        if (header && header.classList.contains('node-header')) {
                            const toggleBtn = header.querySelector('.toggle-btn');
                            if (toggleBtn) toggleBtn.textContent = '▼';
                        }
                        parent = parent.parentElement;
                    }
                    // Add highlight class
                    node.classList.add('search-highlight');
                } else {
                    node.style.display = 'none';
                    node.classList.remove('search-highlight');
                }
            });
        });
    }
    
    // Initialize - collapse all nodes by default
    document.querySelectorAll('.node-children').forEach(children => {
        children.style.display = 'none';
    });
    
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
            console.log('Collection phase changed to:', this.value);
        });
    }
});
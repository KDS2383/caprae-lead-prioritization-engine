document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('search-form');
    const loader = document.getElementById('loader');
    const resultsBody = document.getElementById('results-body');

    searchForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission

        // Get user input
        const industry = document.getElementById('industry').value;
        const location = document.getElementById('location').value;

        // Show loader and clear previous results
        loader.style.display = 'block';
        resultsBody.innerHTML = '';

        try {
            // Make an API call to our Flask backend
            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ industry, location }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const leads = await response.json();

            // Hide loader
            loader.style.display = 'none';
            
            // Populate the table with prioritized leads
            populateTable(leads);

        } catch (error) {
            // Hide loader and show an error
            loader.style.display = 'none';
            resultsBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: #ff6b6b;">An error occurred. Please try again.</td></tr>`;
            console.error('Error fetching leads:', error);
        }
    });

    function populateTable(leads) {
        if (leads.length === 0) {
            // Update colspan to 8
            resultsBody.innerHTML = `<tr><td colspan="8" style="text-align: center;">No leads found for this query.</td></tr>`;
            return;
        }

        leads.forEach((lead, index) => {
            const row = document.createElement('tr');
            
            // Added a clickable link for the URL
            row.innerHTML = `
                <td>${index + 1}</td>
                <td class="score">${lead.lead_score}</td>
                <td>${lead.name}</td>
                <td><a href="${lead.url}" target="_blank">${lead.url}</a></td>
                <td>$${lead.revenue_mil}M</td>
                <td>${lead.employees}</td>
                <td>${lead.bbb_rating}</td>
            `;
            
            resultsBody.appendChild(row);
        });
    }


});
(() => {
  'use strict';

  // --- Theme Toggle (Light / Dark Mode) ---
  const themeToggle = document.getElementById('themeToggle');
  const savedTheme = localStorage.getItem('aegis_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);

  themeToggle?.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', nextTheme);
    localStorage.setItem('aegis_theme', nextTheme);
  });

  // --- DOM Elements ---
  const loanForm = document.getElementById('loanForm');
  const evaluateBtn = document.getElementById('evaluateBtn');
  const evalSpinner = document.getElementById('evalSpinner');

  // Input Fields
  const personAge = document.getElementById('person_age');
  const personIncome = document.getElementById('person_income');
  const personHome = document.getElementById('person_home_ownership');
  const personEmp = document.getElementById('person_emp_length');
  const loanIntent = document.getElementById('loan_intent');
  const loanGrade = document.getElementById('loan_grade');
  const loanAmnt = document.getElementById('loan_amnt');
  const loanRate = document.getElementById('loan_int_rate');
  const loanPercentIncome = document.getElementById('loan_percent_income');
  const cbDefault = document.getElementById('cb_person_default_on_file');
  const cbHistLen = document.getElementById('cb_person_cred_hist_length');

  // Ratio Elements
  const ratioValue = document.getElementById('ratioValue');
  const ratioFill = document.getElementById('ratioFill');

  // Results & Scale Elements
  const gaugeProgress = document.getElementById('gaugeProgress');
  const gaugeProbability = document.getElementById('gaugeProbability');
  const decisionStatusBadge = document.getElementById('decisionStatusBadge');
  const verdictTitle = document.getElementById('verdictTitle');
  const verdictDesc = document.getElementById('verdictDesc');
  const cardTier = document.getElementById('cardTier');

  // Attribution Lists
  const riskDriversList = document.getElementById('riskDriversList');
  const protectiveFactorsList = document.getElementById('protectiveFactorsList');

  // Simulator Elements
  const simAmnt = document.getElementById('simAmnt');
  const simIncome = document.getElementById('simIncome');
  const simRate = document.getElementById('simRate');
  const simEmp = document.getElementById('simEmp');
  const simAmntVal = document.getElementById('simAmntVal');
  const simIncomeVal = document.getElementById('simIncomeVal');
  const simRateVal = document.getElementById('simRateVal');
  const simEmpVal = document.getElementById('simEmpVal');
  const simScoreDisplay = document.getElementById('simScoreDisplay');
  const simVerdictTag = document.getElementById('simVerdictTag');

  let currentThreshold = 0.5683;

  // --- Tab Navigation ---
  document.querySelectorAll('.nav-link').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-link').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      document.getElementById(targetId)?.classList.add('active');

      if (targetId === 'tab-simulator') {
        runSimulation();
      }
    });
  });

  // --- Auto-calculate Loan-to-Income (DTI) ---
  function updateRatio() {
    const income = parseFloat(personIncome.value) || 1;
    const amnt = parseFloat(loanAmnt.value) || 0;
    const ratio = Math.min(1.0, Math.max(0.01, amnt / income));
    loanPercentIncome.value = ratio.toFixed(2);
    ratioValue.textContent = ratio.toFixed(2);
    const pct = Math.min(100, Math.round(ratio * 100));
    ratioFill.style.width = `${pct}%`;
    ratioFill.style.backgroundColor = ratio > 0.4 ? 'var(--accent-risk)' : (ratio > 0.25 ? '#f59e0b' : 'var(--accent-safe)');
  }

  personIncome.addEventListener('input', updateRatio);
  loanAmnt.addEventListener('input', updateRatio);
  updateRatio();

  // --- Quick Preset Profiles ---
  document.getElementById('presetPrime')?.addEventListener('click', () => {
    personAge.value = 35;
    personIncome.value = 95000;
    personHome.value = 'MORTGAGE';
    personEmp.value = 8.0;
    loanIntent.value = 'HOMEIMPROVEMENT';
    loanGrade.value = 'A';
    loanAmnt.value = 10000;
    loanRate.value = 7.5;
    cbDefault.value = 'N';
    cbHistLen.value = 10;
    updateRatio();
  });

  document.getElementById('presetSubprime')?.addEventListener('click', () => {
    personAge.value = 22;
    personIncome.value = 24000;
    personHome.value = 'RENT';
    personEmp.value = 1.0;
    loanIntent.value = 'PERSONAL';
    loanGrade.value = 'D';
    loanAmnt.value = 14000;
    loanRate.value = 18.5;
    cbDefault.value = 'Y';
    cbHistLen.value = 2;
    updateRatio();
  });

  document.getElementById('presetBorderline')?.addEventListener('click', () => {
    personAge.value = 28;
    personIncome.value = 52000;
    personHome.value = 'RENT';
    personEmp.value = 3.5;
    loanIntent.value = 'DEBTCONSOLIDATION';
    loanGrade.value = 'C';
    loanAmnt.value = 16000;
    loanRate.value = 13.2;
    cbDefault.value = 'N';
    cbHistLen.value = 5;
    updateRatio();
  });

  // --- Check API Health ---
  fetch('/health')
    .then(res => res.json())
    .then(data => {
      if (data.optimal_threshold) {
        currentThreshold = data.optimal_threshold;
      }
    })
    .catch(() => {
      const statusText = document.getElementById('apiStatusText');
      if (statusText) statusText.textContent = 'Offline';
      const dot = document.querySelector('.status-dot');
      if (dot) dot.style.backgroundColor = 'var(--accent-risk)';
    });

  // --- Loan Evaluation Submission ---
  loanForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    evalSpinner.style.display = 'inline-block';
    evaluateBtn.disabled = true;

    const payload = {
      person_age: parseInt(personAge.value, 10),
      person_income: parseFloat(personIncome.value),
      person_home_ownership: personHome.value,
      person_emp_length: parseFloat(personEmp.value),
      loan_intent: loanIntent.value,
      loan_grade: loanGrade.value,
      loan_amnt: parseFloat(loanAmnt.value),
      loan_int_rate: parseFloat(loanRate.value),
      loan_percent_income: parseFloat(loanPercentIncome.value),
      cb_person_default_on_file: cbDefault.value,
      cb_person_cred_hist_length: parseInt(cbHistLen.value, 10)
    };

    try {
      const explainRes = await fetch('/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!explainRes.ok) {
        throw new Error(`API returned HTTP ${explainRes.status}`);
      }

      const explainData = await explainRes.json();
      renderDecision(explainData);

    } catch (err) {
      alert(`Underwriting Service: ${err.message}. Ensure the FastAPI server is running on localhost:8000.`);
    } finally {
      evalSpinner.style.display = 'none';
      evaluateBtn.disabled = false;
    }
  });

  // --- Render Decision Output ---
  function renderDecision(data) {
    const prob = data.default_probability;
    const isHighRisk = data.risk_verdict === 'High Risk';
    const probPct = (prob * 100).toFixed(1);

    // Number readout & linear scale
    gaugeProbability.textContent = `${probPct}%`;
    gaugeProgress.style.width = `${Math.min(100, Math.max(0, prob * 100))}%`;
    gaugeProgress.style.backgroundColor = isHighRisk ? 'var(--accent-risk)' : 'var(--accent-safe)';

    // Status Badge & Verdict
    if (isHighRisk) {
      decisionStatusBadge.className = 'status-badge badge-danger';
      decisionStatusBadge.textContent = 'Declined \u2022 High Risk';
      verdictTitle.textContent = 'High Default Risk';
      verdictTitle.style.color = 'var(--accent-risk)';
      verdictDesc.textContent = `Calculated default probability is ${probPct}%. Key risk contributors identified below.`;
      cardTier.textContent = 'Declined';
      cardTier.style.color = 'var(--accent-risk)';
    } else {
      decisionStatusBadge.className = 'status-badge badge-success';
      decisionStatusBadge.textContent = 'Approved \u2022 Low Risk';
      verdictTitle.textContent = 'Eligible for Approval';
      verdictTitle.style.color = 'var(--accent-safe)';
      verdictDesc.textContent = `Calculated default probability is ${probPct}%. Favorable financial indicators supported approval.`;
      cardTier.textContent = 'Approved';
      cardTier.style.color = 'var(--accent-safe)';
    }

    // Risk Drivers (Red)
    riskDriversList.innerHTML = '';
    if (data.top_risk_drivers && data.top_risk_drivers.length > 0) {
      const maxImpact = Math.max(...data.top_risk_drivers.map(d => d.impact_value), 0.05);
      data.top_risk_drivers.forEach(driver => {
        const barPct = Math.min(100, Math.round((driver.impact_value / maxImpact) * 100));
        const item = document.createElement('div');
        item.className = 'factor-item';
        item.innerHTML = `
          <div class="factor-row">
            <span>${driver.display_name}</span>
            <span class="text-risk">+${driver.impact_value.toFixed(3)} SHAP</span>
          </div>
          <div class="factor-bar-track">
            <div class="factor-bar-fill danger" style="width: ${barPct}%;"></div>
          </div>
        `;
        riskDriversList.appendChild(item);
      });
    } else {
      riskDriversList.innerHTML = '<div class="empty-note">No notable risk escalators found.</div>';
    }

    // Protective Factors (Green)
    protectiveFactorsList.innerHTML = '';
    if (data.top_protective_factors && data.top_protective_factors.length > 0) {
      const maxImpact = Math.max(...data.top_protective_factors.map(d => d.impact_value), 0.05);
      data.top_protective_factors.forEach(driver => {
        const barPct = Math.min(100, Math.round((driver.impact_value / maxImpact) * 100));
        const item = document.createElement('div');
        item.className = 'factor-item';
        item.innerHTML = `
          <div class="factor-row">
            <span>${driver.display_name}</span>
            <span class="text-safe">-${driver.impact_value.toFixed(3)} SHAP</span>
          </div>
          <div class="factor-bar-track">
            <div class="factor-bar-fill success" style="width: ${barPct}%;"></div>
          </div>
        `;
        protectiveFactorsList.appendChild(item);
      });
    } else {
      protectiveFactorsList.innerHTML = '<div class="empty-note">No notable mitigating factors found.</div>';
    }
  }

  // --- Real-Time Sensitivity Simulator ---
  let simDebounceTimer = null;

  function runSimulation() {
    clearTimeout(simDebounceTimer);
    simDebounceTimer = setTimeout(async () => {
      const amnt = parseFloat(simAmnt.value);
      const inc = parseFloat(simIncome.value);
      const rate = parseFloat(simRate.value);
      const emp = parseFloat(simEmp.value);

      simAmntVal.textContent = `$${amnt.toLocaleString()}`;
      simIncomeVal.textContent = `$${inc.toLocaleString()}`;
      simRateVal.textContent = `${rate.toFixed(1)}%`;
      simEmpVal.textContent = `${emp.toFixed(1)} Years`;

      const simPayload = {
        person_age: 30,
        person_income: inc,
        person_home_ownership: 'RENT',
        person_emp_length: emp,
        loan_intent: 'PERSONAL',
        loan_grade: rate > 15 ? 'D' : (rate > 10 ? 'C' : 'B'),
        loan_amnt: amnt,
        loan_int_rate: rate,
        loan_percent_income: parseFloat((amnt / inc).toFixed(2)),
        cb_person_default_on_file: 'N',
        cb_person_cred_hist_length: 5
      };

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(simPayload)
        });

        if (res.ok) {
          const simData = await res.json();
          const probPct = (simData.default_probability * 100).toFixed(1);
          simScoreDisplay.textContent = `${probPct}%`;

          if (simData.default_prediction === 1) {
            simScoreDisplay.style.color = 'var(--accent-risk)';
            simVerdictTag.textContent = 'High Risk';
            simVerdictTag.style.color = 'var(--accent-risk)';
          } else {
            simScoreDisplay.style.color = 'var(--accent-safe)';
            simVerdictTag.textContent = 'Low Risk';
            simVerdictTag.style.color = 'var(--accent-safe)';
          }
        }
      } catch (e) {
        // quiet ignore during rapid sliding
      }
    }, 120);
  }

  [simAmnt, simIncome, simRate, simEmp].forEach(input => {
    input?.addEventListener('input', runSimulation);
  });

})();

// Slider definitions (label, range, default)
const FIELDS = [
    { id: "Pregnancies", label: "Pregnancies", min: 0, max: 17, val: 1 },
    { id: "Glucose", label: "Glucose", unit: "mg/dL", min: 40, max: 250, val: 120 },
    { id: "BloodPressure", label: "Blood pressure", unit: "mm Hg", min: 30, max: 130, val: 70 },
    { id: "SkinThickness", label: "Skin thickness", unit: "mm", min: 5, max: 100, val: 25, optional: true },
    { id: "Insulin", label: "Insulin", unit: "µU/mL", min: 10, max: 850, val: 80, optional: true },
    { id: "BMI", label: "BMI", min: 15, max: 70, step: 0.1, val: 28 },
    { id: "DiabetesPedigreeFunction", label: "Family history score", min: 0.05, max: 2.5, step: 0.01, val: 0.4 },
    { id: "Age", label: "Age", unit: "yrs", min: 18, max: 90, val: 30 },
];

// Build the sliders
const grid = document.getElementById("fields");
FIELDS.forEach((f) => {
    const d = document.createElement("div");
    d.className = "field";
    d.innerHTML = `
    <div class="field-top">
      <label for="${f.id}">${f.label}${f.unit ? ` <small>${f.unit}</small>` : ""}</label>
      <output>${f.val}</output>
    </div>
    <input type="range" id="${f.id}" min="${f.min}" max="${f.max}" step="${f.step || 1}" value="${f.val}">
    ${f.optional ? `<label class="unknown"><input type="checkbox" id="${f.id}-unk"> Unknown</label>` : ""}`;
    grid.appendChild(d);

    const input = d.querySelector("input[type=range]");
    const out = d.querySelector("output");
    const paint = () => {
        out.textContent = input.value;
        input.style.setProperty("--p", ((input.value - f.min) / (f.max - f.min)) * 100 + "%");
    };
    input.addEventListener("input", paint);
    paint();

    if (f.optional) {
        d.querySelector(`#${f.id}-unk`).addEventListener("change", (e) => {
            input.disabled = e.target.checked;
            out.textContent = e.target.checked ? "unknown" : input.value;
        });
    }
});

// Send values to Flask and show the result
const btn = document.getElementById("predict-btn");
btn.addEventListener("click", async () => {
    const payload = {};
    FIELDS.forEach((f) => {
        const el = document.getElementById(f.id);
        payload[f.id] = el.disabled ? null : parseFloat(el.value);
    });
    btn.classList.add("loading");
    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Server error");
        showResult(data);
    } catch (err) {
        alert(err.message);
    }
    btn.classList.remove("loading");
});

function showResult(d) {
    document.getElementById("bar").style.strokeDashoffset = 326.7 * (1 - d.probability);
    document.getElementById("pct").textContent = Math.round(d.probability * 100) + "%";
    const v = document.getElementById("verdict");
    v.textContent = d.high_risk ? "High risk" : "Low risk";
    v.className = "verdict " + (d.high_risk ? "high" : "low");
    document.getElementById("advice").textContent = d.high_risk
        ? "These values suggest elevated risk. A medical check-up is recommended."
        : "These values suggest lower risk. Maintain regular health check-ups.";
}

// Fade sections in as they scroll into view
const io = new IntersectionObserver(
    (entries) => entries.forEach((e) => e.isIntersecting && e.target.classList.add("in")),
    { threshold: 0.12 }
);
document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
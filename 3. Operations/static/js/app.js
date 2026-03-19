document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("predictBtn");
  const kdInput = document.getElementById("kd");
  const matchesInput = document.getElementById("matchesPlayed");
  const headshotInput = document.getElementById("headshotPct");
  const dmgInput = document.getElementById("dmgRnd");
  const rankInput = document.getElementById("rank");
  const statusEl = document.getElementById("status");
  const resultsPanel = document.getElementById("resultsPanel");
  const resultEl = document.getElementById("result");
  const rankDisplay = document.getElementById("rankDisplay");
  const mmrBadge = document.getElementById("mmrBadge");
  const plotsEl = document.getElementById("plots");
  const rankPopup = document.getElementById("rankPopup");
  const rankPopupText = document.getElementById("rankPopupText");
  const rankPopupImage = document.getElementById("rankPopupImage");
  const closePopupBtn = document.getElementById("closePopupBtn");

  const rankOrder = [
    "Unranked",
    "Iron",
    "Bronze",
    "Silver",
    "Gold",
    "Platinum",
    "Diamond",
    "Ascendant",
    "Immortal",
    "Radiant",
  ];

  const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
  function shiftFromPrediction(p) {
    if (p >= 80) return 2;
    if (p >= 70) return 1;
    if (p >= 45) return 0;
    if (p >= 30) return -1;
    return -2;
  }
  function getPredictedRank(currentRank, predictedWinPct) {
    const i = rankOrder.indexOf(currentRank);
    if (i === -1) return null;
    return rankOrder[
      clamp(i + shiftFromPrediction(predictedWinPct), 0, rankOrder.length - 1)
    ];
  }
  function showRankPopup(currentRank, predictedRank, predictedWinPct) {
    rankPopupText.textContent = `Current rank: ${currentRank} | Predicted rank: ${predictedRank} | Predicted win%: ${predictedWinPct.toFixed(2)}%`;
    rankPopupImage.src = `/static/images/ranks/${predictedRank.toLowerCase()}.png`;
    rankPopup.style.display = "flex";
  }

  closePopupBtn.addEventListener(
    "click",
    () => (rankPopup.style.display = "none"),
  );
  rankPopup.addEventListener("click", (e) => {
    if (e.target === rankPopup) rankPopup.style.display = "none";
  });

  btn.addEventListener("click", async () => {
    statusEl.style.display = "none";
    statusEl.textContent = "";

    const kd = Number(kdInput.value);
    const matchesPlayed = Number(matchesInput.value);
    const headshotPct = Number(headshotInput.value);
    const dmgRnd = Number(dmgInput.value);
    const rank = rankInput.value || "Unranked";

    if (
      kdInput.value === "" ||
      matchesInput.value === "" ||
      headshotInput.value === "" ||
      dmgInput.value === ""
    ) {
      statusEl.textContent = "Please fill in all required feature fields.";
      statusEl.style.display = "block";
      return;
    }

    const payload = {
      features: [kd, matchesPlayed, headshotPct, dmgRnd],
      rank,
    };

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();

      if (!response.ok) {
        statusEl.textContent =
          data.error || `Request failed (${response.status})`;
        statusEl.style.display = "block";
        return;
      }

      resultsPanel.style.display = "block";
      resultEl.textContent = `${data.predicted_value.toFixed(4)}%`;
      rankDisplay.textContent = `Rank: ${rank}`;

      mmrBadge.textContent = `${data.mmr_label} — ${data.mmr_detail}`;
      mmrBadge.className = "mmr-badge";
      if (data.mmr_label === "High MMR") mmrBadge.classList.add("mmr-high");
      else if (data.mmr_label === "Low MMR") mmrBadge.classList.add("mmr-low");
      else mmrBadge.classList.add("mmr-medium");

      plotsEl.innerHTML = "";
      (data.plots || []).forEach((plot) => {
        const card = document.createElement("div");
        card.className = "plot-card";
        const img = document.createElement("img");
        img.src = `data:image/png;base64,${plot.image}`;
        img.alt = `Plot for ${plot.feature}`;
        card.appendChild(img);
        plotsEl.appendChild(card);
      });

      if (rank !== "Unranked") {
        const predictedRank = getPredictedRank(rank, data.predicted_value);
        if (predictedRank)
          showRankPopup(rank, predictedRank, data.predicted_value);
      }
    } catch (err) {
      statusEl.textContent = `Network/server error: ${String(err)}`;
      statusEl.style.display = "block";
    }
  });
});

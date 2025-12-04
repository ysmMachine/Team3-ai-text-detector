let currentStage = 0;

function updateLoadingStage(stage) {
    const stages = ['stage1', 'stage2', 'stage3'];
    for (let i = 0; i < stage; i++) {
        document.getElementById(stages[i]).className = 'loading-stage complete';
    }
    if (stage < stages.length) {
        document.getElementById(stages[stage]).className = 'loading-stage active';
    }
}

function getColorForScore(score, isAI) {
    const absScore = Math.abs(score);
    const intensity = Math.min(absScore * 2, 1);
    if (isAI) {
        const r = 239;
        const g = Math.round(68 + (255 - 68) * (1 - intensity));
        const b = Math.round(68 + (255 - 68) * (1 - intensity));
        return `rgba(${r}, ${g}, ${b}, ${0.2 + intensity * 0.5})`;
    } else {
        const r = Math.round(59 + (255 - 59) * (1 - intensity));
        const g = Math.round(130 + (255 - 130) * (1 - intensity));
        const b = 246;
        return `rgba(${r}, ${g}, ${b}, ${0.2 + intensity * 0.5})`;
    }
}

function highlightWithLIME(text, wordImportance) {
    if (!wordImportance || Object.keys(wordImportance).length === 0) {
        return { highlighted: text, count: 0, avgScore: 0 };
    }

    let highlighted = text;
    let matchCount = 0;
    let totalScore = 0;

    const sortedWords = Object.entries(wordImportance).sort((a, b) =>
        Math.abs(b[1]) - Math.abs(a[1])
    );

    sortedWords.forEach(([word, score]) => {
        if (Math.abs(score) < 0.001) return;

        const isAI = score > 0;
        const className = isAI ? 'ai-keyword' : 'human-keyword';
        const bgColor = getColorForScore(score, isAI);
        const direction = isAI ? 'AI' : 'Human';
        const sign = score > 0 ? '+' : '';

        const regex = new RegExp(`\\b(${word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})\\b`, 'gi');

        const matches = text.match(regex);
        if (matches) {
            matchCount += matches.length;
            totalScore += Math.abs(score);

            highlighted = highlighted.replace(regex, (match) => {
                return `<span class="${className}" style="background: ${bgColor};">${match}<span class="tooltip">${direction} 기여도: ${sign}${score.toFixed(5)}</span></span>`;
            });
        }
    });

    const avgScore = matchCount > 0 ? (totalScore / matchCount).toFixed(3) : 0;
    return { highlighted, count: matchCount, avgScore };
}

async function predict() {
    const text = document.getElementById('inputText').value;
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');
    const analyzeBtn = document.getElementById('analyzeBtn');

    if (!text.trim()) {
        alert("⚠️ 분석할 텍스트를 입력해주세요.");
        return;
    }

    currentStage = 0;
    updateLoadingStage(0);

    loadingDiv.style.display = 'block';
    resultDiv.style.display = 'none';
    analyzeBtn.disabled = true;

    try {
        updateLoadingStage(0);
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        updateLoadingStage(1);
        const data = await response.json();

        updateLoadingStage(2);
        await new Promise(resolve => setTimeout(resolve, 300));

        const predictionEl = document.getElementById('prediction');
        predictionEl.innerText = data.prediction;
        predictionEl.className = 'prediction-badge ' + (data.ai_probability > 0.5 ? 'ai' : 'human');

        document.getElementById('confidence').innerText = data.confidence;

        const { highlighted, count, avgScore } = highlightWithLIME(text, data.word_importance);
        document.getElementById('highlightedText').innerHTML = highlighted;

        const totalWords = text.trim().split(/\s+/).length;
        document.getElementById('keywordCount').innerText = count;
        document.getElementById('totalWords').innerText = totalWords;
        document.getElementById('avgImportance').innerText = avgScore;

        updateLoadingStage(3);
        loadingDiv.style.display = 'none';
        resultDiv.style.display = 'block';

    } catch (error) {
        console.error('Error:', error);
        loadingDiv.style.display = 'none';
        alert("❌ 예측 중 오류가 발생했습니다. 서버가 실행 중인지 확인해주세요.");
    } finally {
        analyzeBtn.disabled = false;
    }
}

document.getElementById('inputText').addEventListener('keydown', function (e) {
    if (e.ctrlKey && e.key === 'Enter') {
        predict();
    }
});

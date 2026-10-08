const fileInput = document.getElementById("file-input");
const fileName = document.getElementById("file-name");
const analyzeButton = document.getElementById("analyze-button");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const prediction = document.getElementById("prediction");

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];

    if (!file) {
        fileName.textContent = "No file selected";
        return;
    }

    fileName.textContent = file.name;
});

analyzeButton.addEventListener("click", async () => {
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an audio or video file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    analyzeButton.disabled = true;
    loading.classList.remove("hidden");
    result.classList.add("hidden");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Prediction request failed.");
        }

        const data = await response.json();

        prediction.textContent = formatReciterName(data.prediction);
        result.classList.remove("hidden");
    } catch (error) {
        console.error(error);
        alert("An error occurred while analyzing the file.");
    } finally {
        analyzeButton.disabled = false;
        loading.classList.add("hidden");
    }
});

function formatReciterName(name) {
    if (name === "Unknown") {
        return "Unknown Reciter";
    }

    return name
        .replace(/^\d+_/, "")
        .split("_")
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");
}

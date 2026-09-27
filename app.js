"use strict";

/* =========================================================
   DIABETIC RETINOPATHY - APP.JS
   ========================================================= */

const state = {
    uploadedFile: null,
    trainChart: null,
    lossChart: null
};


/* =========================================================
   HELPERS
   ========================================================= */

const $ = (id) => document.getElementById(id);

function toast(message, type = "success") {
    const el = $("toast");

    if (!el) {
        alert(message);
        return;
    }

    el.textContent = (type === "success" ? "✅ " : "❌ ") + message;
    el.className = `show ${type}`;

    setTimeout(() => {
        el.className = "";
    }, 3500);
}


function setLoading(id, loading, normalText) {

    const btn = $(id);

    if (!btn) return;

    btn.disabled = loading;

    if (loading) {
        btn.innerHTML = "⏳ Processing...";
    } else {
        btn.innerHTML = normalText;
    }
}


function markDone(id) {

    const dot = $(id);

    if (!dot) return;

    dot.classList.remove("active");
    dot.classList.add("done");
}


function markActive(id) {

    const dot = $(id);

    if (!dot) return;

    dot.classList.add("active");
}


/* =========================================================
   TAB NAVIGATION
   ========================================================= */

function switchTab(tabId) {

    document.querySelectorAll(".tab-panel").forEach(panel => {
        panel.classList.remove("active");
    });

    document.querySelectorAll(".nav-item").forEach(item => {
        item.classList.remove("active");
    });

    const panel = $(tabId);

    if (panel) {
        panel.classList.add("active");
    }

    const nav = document.querySelector(
        `.nav-item[data-tab="${tabId}"]`
    );

    if (nav) {
        nav.classList.add("active");
    }
}


document.querySelectorAll(".nav-item[data-tab]").forEach(item => {

    item.addEventListener("click", () => {
        switchTab(item.dataset.tab);
    });

});


/* =========================================================
   IMAGE UPLOAD
   ========================================================= */

const uploadZone = $("uploadZone");
const fileInput = $("fileInput");


if (uploadZone && fileInput) {

    uploadZone.addEventListener("click", () => {
        fileInput.click();
    });


    uploadZone.addEventListener("dragover", (event) => {

        event.preventDefault();

        uploadZone.classList.add("dragover");

    });


    uploadZone.addEventListener("dragleave", () => {

        uploadZone.classList.remove("dragover");

    });


    uploadZone.addEventListener("drop", (event) => {

        event.preventDefault();

        uploadZone.classList.remove("dragover");

        const file = event.dataTransfer.files[0];

        if (file) {
            handleUpload(file);
        }

    });


    fileInput.addEventListener("change", () => {

        const file = fileInput.files[0];

        if (file) {
            handleUpload(file);
        }

    });

}


/* =========================================================
   HANDLE IMAGE
   ========================================================= */

async function handleUpload(file) {

    console.log("Selected file:", file.name);

    if (!file.type.startsWith("image/")) {

        toast(
            "Please select a JPG, PNG, BMP or TIFF image",
            "error"
        );

        return;
    }


    setLoading(
        "uploadBtn",
        true,
        "📂 Choose Image"
    );


    /*
       IMPORTANT:
       Show image immediately in browser.
       This means preview works even if Flask upload
       has a problem.
    */

    const localPreview = URL.createObjectURL(file);

    const previewImg = $("previewImg");

    if (previewImg) {
        previewImg.src = localPreview;
    }


    /*
       Show file information immediately
    */

    if ($("previewBox")) {
        $("previewBox").style.display = "block";
    }


    if ($("imgFilename")) {
        $("imgFilename").textContent = file.name;
    }


    if ($("imgSize")) {

        const sizeKB =
            (file.size / 1024).toFixed(1);

        $("imgSize").textContent =
            `${sizeKB} KB`;
    }


    if ($("imgType")) {

        $("imgType").textContent =
            file.type
                .split("/")
                .pop()
                .toUpperCase();
    }


    /*
       Get image dimensions
    */

    const image = new Image();

    image.onload = () => {

        if ($("imgDims")) {

            $("imgDims").textContent =
                `${image.width} × ${image.height}`;
        }

    };

    image.src = localPreview;


    /*
       Now send image to Flask
    */

    const formData = new FormData();

    formData.append("file", file);


    try {

        const response = await fetch(
            "/api/upload",
            {
                method: "POST",
                body: formData
            }
        );


        const text = await response.text();

        let data;

        try {

            data = JSON.parse(text);

        } catch {

            throw new Error(
                "Server returned an invalid response. Check Flask terminal."
            );

        }


        if (!response.ok) {

            throw new Error(
                data.error || "Upload failed"
            );

        }


        /*
           If server sends preview,
           use server preview.
        */

        if (data.preview_b64) {

            previewImg.src =
                `data:image/jpeg;base64,${data.preview_b64}`;
        }


        state.uploadedFile =
            data.filename || file.name;


        markDone("dot-upload");

        toast(
            "Image loaded successfully"
        );


        console.log(
            "Upload response:",
            data
        );

    }


    catch (error) {

        console.error(
            "UPLOAD ERROR:",
            error
        );


        /*
           Local preview is already visible.
           So don't hide it if backend has an error.
        */

        toast(
            "Image preview loaded, but server upload failed: " +
            error.message,
            "error"
        );

    }


    finally {

        setLoading(
            "uploadBtn",
            false,
            "📂 Choose Image"
        );

    }

}


/* =========================================================
   PREPROCESSING
   ========================================================= */

const preprocessBtn = $("preprocessBtn");


if (preprocessBtn) {

    preprocessBtn.addEventListener(
        "click",
        async () => {

            markActive("dot-pre");

            setLoading(
                "preprocessBtn",
                true,
                "⚙️ Run Preprocessing"
            );


            try {

                const response =
                    await fetch(
                        "/api/preprocess",
                        {
                            method: "POST"
                        }
                    );


                const data =
                    await response.json();


                if (!response.ok) {

                    throw new Error(
                        data.error ||
                        "Preprocessing failed"
                    );

                }


                /*
                   Original image
                */

                if ($("origImg") &&
                    data.original_b64) {

                    $("origImg").src =
                        `data:image/jpeg;base64,${data.original_b64}`;
                }


                /*
                   Processed image
                */

                if ($("procImg") &&
                    data.processed_b64) {

                    $("procImg").src =
                        `data:image/jpeg;base64,${data.processed_b64}`;
                }


                if ($("preCompareCard")) {

                    $("preCompareCard").style.display =
                        "block";
                }


                /*
                   Statistics
                */

                if (data.stats) {

                    const s = data.stats;


                    if ($("statW"))
                        $("statW").textContent =
                            s.width ?? "—";


                    if ($("statH"))
                        $("statH").textContent =
                            s.height ?? "—";


                    if ($("statOM"))
                        $("statOM").textContent =
                            Number(
                                s.original_mean ?? 0
                            ).toFixed(1);


                    if ($("statPM"))
                        $("statPM").textContent =
                            Number(
                                s.processed_mean ?? 0
                            ).toFixed(1);


                    if ($("statOS"))
                        $("statOS").textContent =
                            Number(
                                s.original_std ?? 0
                            ).toFixed(1);


                    if ($("statPS"))
                        $("statPS").textContent =
                            Number(
                                s.processed_std ?? 0
                            ).toFixed(1);

                }


                if ($("preStats")) {

                    $("preStats").style.display =
                        "block";
                }


                markDone("dot-pre");

                toast(
                    "Preprocessing completed"
                );

            }


            catch (error) {

                console.error(
                    "PREPROCESS ERROR:",
                    error
                );

                toast(
                    error.message,
                    "error"
                );

            }


            finally {

                setLoading(
                    "preprocessBtn",
                    false,
                    "⚙️ Run Preprocessing"
                );

            }

        }
    );

}


/* =========================================================
   FEATURE EXTRACTION
   ========================================================= */

const extractBtn = $("extractBtn");


if (extractBtn) {

    extractBtn.addEventListener(
        "click",
        async () => {

            markActive("dot-feat");

            setLoading(
                "extractBtn",
                true,
                "🔬 Extract Features"
            );


            try {

                const response =
                    await fetch(
                        "/api/extract",
                        {
                            method: "POST"
                        }
                    );


                const data =
                    await response.json();


                if (!response.ok) {

                    throw new Error(
                        data.error ||
                        "Feature extraction failed"
                    );

                }


                const grid = $("fmGrid");

                if (grid) {

                    grid.innerHTML = "";

                    (data.feature_maps || [])
                        .forEach((image, index) => {

                            const div =
                                document.createElement("div");

                            div.className =
                                "fm-item";


                            div.innerHTML = `
                                <img
                                    src="data:image/jpeg;base64,${image}"
                                    alt="Feature Map ${index + 1}"
                                />
                            `;


                            grid.appendChild(div);

                        });

                }


                if ($("featureDesc")) {

                    $("featureDesc").textContent =
                        data.description ||
                        "Feature maps extracted successfully.";

                }


                if ($("fmSection")) {

                    $("fmSection").style.display =
                        "block";

                }


                markDone("dot-feat");

                toast(
                    "Feature extraction completed"
                );

            }


            catch (error) {

                console.error(error);

                toast(
                    error.message,
                    "error"
                );

            }


            finally {

                setLoading(
                    "extractBtn",
                    false,
                    "🔬 Extract Features"
                );

            }

        }
    );

}


/* =========================================================
   TRAINING
   ========================================================= */

const trainBtn = $("trainBtn");


if (trainBtn) {

    trainBtn.addEventListener(
        "click",
        async () => {

            const epochs =
                parseInt(
                    $("epochsInput")?.value
                ) || 10;


            markActive("dot-train");


            setLoading(
                "trainBtn",
                true,
                "🚀 Start Training"
            );


            if ($("epochTableBody")) {
                $("epochTableBody").innerHTML = "";
            }


            try {

                const response =
                    await fetch(
                        "/api/train",
                        {
                            method: "POST",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body: JSON.stringify({
                                epochs: epochs
                            })
                        }
                    );


                const data =
                    await response.json();


                if (!response.ok) {

                    throw new Error(
                        data.error ||
                        "Training failed"
                    );

                }


                const history =
                    data.history;


                for (
                    let i = 0;
                    i < history.accuracy.length;
                    i++
                ) {

                    const row =
                        document.createElement("tr");


                    row.innerHTML = `
                        <td>${i + 1}</td>

                        <td>
                            ${Number(
                                history.loss[i]
                            ).toFixed(4)}
                        </td>

                        <td>
                            ${Number(
                                history.val_loss[i]
                            ).toFixed(4)}
                        </td>

                        <td>
                            ${(history.accuracy[i] * 100)
                                .toFixed(2)}%
                        </td>

                        <td>
                            ${(history.val_accuracy[i] * 100)
                                .toFixed(2)}%
                        </td>
                    `;


                    $("epochTableBody")
                        .appendChild(row);


                    if ($("epochFill")) {

                        $("epochFill").style.width =
                            `${((i + 1) /
                                history.accuracy.length) * 100}%`;

                    }

                }


                buildTrainCharts(history);


                if ($("trainSection")) {

                    $("trainSection").style.display =
                        "block";

                }


                markDone("dot-train");

                toast(
                    "Training completed"
                );

            }


            catch (error) {

                console.error(error);

                toast(
                    error.message,
                    "error"
                );

            }


            finally {

                setLoading(
                    "trainBtn",
                    false,
                    "🚀 Start Training"
                );

            }

        }
    );

}


/* =========================================================
   CHARTS
   ========================================================= */

function buildTrainCharts(history) {

    if (typeof Chart === "undefined") {

        console.warn(
            "Chart.js not loaded"
        );

        return;
    }


    destroyCharts();


    const labels =
        history.accuracy.map(
            (_, index) =>
                `Epoch ${index + 1}`
        );


    state.trainChart =
        new Chart(
            $("accChart"),
            {
                type: "line",

                data: {

                    labels: labels,

                    datasets: [

                        {
                            label:
                                "Training Accuracy",

                            data:
                                history.accuracy
                                    .map(
                                        v =>
                                            v * 100
                                    )
                        },

                        {
                            label:
                                "Validation Accuracy",

                            data:
                                history.val_accuracy
                                    .map(
                                        v =>
                                            v * 100
                                    )
                        }

                    ]

                },

                options: {
                    responsive: true
                }

            }
        );


    state.lossChart =
        new Chart(
            $("lossChart"),
            {
                type: "line",

                data: {

                    labels: labels,

                    datasets: [

                        {
                            label:
                                "Training Loss",

                            data:
                                history.loss
                        },

                        {
                            label:
                                "Validation Loss",

                            data:
                                history.val_loss
                        }

                    ]

                },

                options: {
                    responsive: true
                }

            }
        );

}


function destroyCharts() {

    if (state.trainChart) {

        state.trainChart.destroy();

        state.trainChart = null;
    }


    if (state.lossChart) {

        state.lossChart.destroy();

        state.lossChart = null;
    }

}


/* =========================================================
   EVALUATION
   ========================================================= */

const evalBtn = $("evalBtn");


if (evalBtn) {

    evalBtn.addEventListener(
        "click",
        async () => {

            markActive("dot-eval");


            setLoading(
                "evalBtn",
                true,
                "📊 Evaluate Model"
            );


            try {

                const response =
                    await fetch(
                        "/api/evaluate",
                        {
                            method: "POST"
                        }
                    );


                const data =
                    await response.json();


                if (!response.ok) {
                    

                    throw new Error(
                        data.error ||
                        "Prediction failed"
                    );

                }


                if ($("predictionSection")) {

                    $("predictionSection")
                        .style.display = "block";

                }


                if ($("predictionResult")) {

                    $("predictionResult")
                        .innerText =
                        data.prediction || "Unknown";

                }


                if ($("predictionConfidence")) {

                    $("predictionConfidence")
                        .innerText =
                        `${data.confidence || 0}%`;

                }


                if ($("predictionMessage")) {

                    $("predictionMessage")
                        .innerText =
                        data.message || "";

                }


                const list =
                    $("probabilityList");


                if (list) {

                    list.innerHTML = "";


                    Object.entries(
                        data.probabilities || {}
                    ).forEach(
                        ([name, probability]) => {

                            const row =
                                document.createElement(
                                    "div"
                                );


                            row.style.marginBottom =
                                "12px";


                            row.innerHTML = `

                                <div style="
                                    display:flex;
                                    justify-content:space-between;
                                    margin-bottom:5px;
                                ">

                                    <span>
                                        ${name}
                                    </span>

                                    <strong>
                                        ${probability}%
                                    </strong>

                                </div>

                                <div style="
                                    width:100%;
                                    height:8px;
                                    background:#17172b;
                                    border-radius:10px;
                                    overflow:hidden;
                                ">

                                    <div style="
                                        width:${probability}%;
                                        height:100%;
                                        background:linear-gradient(
                                            90deg,
                                            #00d4ff,
                                            #00ffcc
                                        );
                                    "></div>

                                </div>

                            `;


                            list.appendChild(row);

                        }
                    );

                }


                toast(
                    "Prediction completed"
                );

            }


            catch (error) {

                console.error(error);

                toast(
                    error.message,
                    "error"
                );

            }


            finally {

                predictBtn.disabled = false;

                predictBtn.innerText =
                    "🔍 Predict DR Severity";

            }

        }
    );

}


/* =========================================================
   INITIALIZE
   ========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        switchTab("tab-upload");

        console.log(
            "Diabetic Retinopathy App loaded successfully."
        );

    }
);
// ═════════════════════════════════════════════
// DIABETIC RETINOPATHY PREDICTION
// ═════════════════════════════════════════════

const predictBtn = document.getElementById("predictBtn");

if (predictBtn) {

    predictBtn.addEventListener("click", async function () {

        predictBtn.disabled = true;
        predictBtn.innerText = "⏳ Predicting...";

        try {

            const response = await fetch("/api/predict", {
                method: "POST"
            });

            const data = await response.json();

            console.log("Prediction response:", data);

            if (!response.ok) {
                alert(data.error || "Prediction failed");
                return;
            }

            const section =
                document.getElementById("predictionResult");

            if (section) {
                section.style.display = "block";
            }

            const result =
                document.getElementById("predictionClass");

            const confidence =
                document.getElementById("predictionConfidence");

            const message =
                document.getElementById("predictionMessage");

            const list =
                document.getElementById("probabilityList");


            if (result) {
                result.innerText = data.prediction;
            }

            if (confidence) {
                confidence.innerText =
                    data.confidence + "%";
            }

            if (message) {
                message.innerText =
                    data.message;
            }


            if (list) {

                list.innerHTML = "";

                Object.entries(data.probabilities)
                    .forEach(([name, probability]) => {

                        const row =
                            document.createElement("div");

                        row.style.marginBottom = "12px";

                        row.innerHTML = `
                            <div style="
                                display:flex;
                                justify-content:space-between;
                                margin-bottom:5px;
                                font-size:0.82rem;
                            ">
                                <span>${name}</span>

                                <strong>
                                    ${probability}%
                                </strong>
                            </div>

                            <div style="
                                width:100%;
                                height:8px;
                                background:#17172b;
                                border-radius:10px;
                                overflow:hidden;
                            ">

                                <div style="
                                    width:${probability}%;
                                    height:100%;
                                    background:linear-gradient(
                                        90deg,
                                        #00d4ff,
                                        #00ffcc
                                    );
                                    border-radius:10px;
                                "></div>

                            </div>
                        `;

                        list.appendChild(row);

                    });
            }

        } catch (error) {

            console.error("Prediction error:", error);

            alert(
                "Could not connect to prediction server."
            );

        } finally {

            predictBtn.disabled = false;

            predictBtn.innerText =
                "🔍 Predict DR Severity";

        }

    });

}
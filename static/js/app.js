const form = document.getElementById("question-form");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send-button");
const chat = document.getElementById("chat");
const welcome = document.getElementById("welcome");


// ============================================================
// Helpers
// ============================================================

function escapeHtml(text) {

    const div = document.createElement("div");

    div.textContent = text ?? "";

    return div.innerHTML;
}


// ============================================================
// Format 3GPP Document Name
// ============================================================

function formatDocumentName(filename) {

    if (!filename) {
        return "Unknown document";
    }

    let name = filename
        .replace(/\.pdf$/i, "")
        .replace(/^ts_/i, "TS ");

    /*
        Example:

        ts_123501v171500p.pdf

        becomes:

        3GPP TS 23.501 • Release 17
    */

    const match = name.match(
        /123(\d{3})v(\d{2})(\d{2})(\d{2})p/i
    );

    if (match) {

        const specification =
            `23.${match[1]}`;

        const release =
            match[2];

        return (
            `3GPP TS ${specification} • Release ${release}`
        );
    }

    return name.toUpperCase();
}


// ============================================================
// Create Source Card
// ============================================================

function createSourceCard(source) {

    const sourceCard =
        document.createElement("div");

    sourceCard.className =
        "source-card";


    // --------------------------------------------------------
    // Document name
    // --------------------------------------------------------

    const sourceName =
        document.createElement("div");

    sourceName.className =
        "source-name";

    sourceName.textContent =
        formatDocumentName(
            source.document
        );

    sourceCard.appendChild(
        sourceName
    );


    // --------------------------------------------------------
    // Page information
    // --------------------------------------------------------

    const sourceMeta =
        document.createElement("div");

    sourceMeta.className =
        "source-meta";

    if (
        source.page !== undefined &&
        source.page !== null
    ) {

        sourceMeta.textContent =
            `Page ${source.page}`;

    } else {

        sourceMeta.textContent =
            "3GPP standards document";
    }

    sourceCard.appendChild(
        sourceMeta
    );


    // --------------------------------------------------------
    // Retrieved PDF / Architecture image
    // --------------------------------------------------------

    if (source.image_url) {

        const imageWrapper =
            document.createElement("div");

        imageWrapper.className =
            "source-image-wrapper";


        const image =
            document.createElement("img");

        image.className =
            "source-image";

        image.src =
            source.image_url;

        image.alt =
            `${formatDocumentName(
                source.document
            )} - Page ${source.page}`;

        image.loading =
            "lazy";


        // ----------------------------------------------------
        // Image error handling
        // ----------------------------------------------------

        image.onerror =
            function () {

                imageWrapper.remove();

            };


        imageWrapper.appendChild(
            image
        );

        sourceCard.appendChild(
            imageWrapper
        );
    }


    return sourceCard;
}


// ============================================================
// Add Chat Message
// ============================================================

function addMessage(
    role,
    text,
    metadata = null
) {

    if (welcome) {
        welcome.remove();
    }


    const wrapper =
        document.createElement("div");

    wrapper.className =
        `message ${role}`;


    const card =
        document.createElement("div");

    card.className =
        "message-card";


    // --------------------------------------------------------
    // Message label
    // --------------------------------------------------------

    const label =
        document.createElement("div");

    label.className =
        "message-label";

    label.textContent =
        role === "user"
            ? "You"
            : "3GPP Assistant";

    card.appendChild(
        label
    );


    // --------------------------------------------------------
    // Message text
    // --------------------------------------------------------

    const messageText =
        document.createElement("div");

    messageText.className =
        "message-text";

    messageText.innerHTML =
        escapeHtml(text);

    card.appendChild(
        messageText
    );


    // ========================================================
    // Assistant Metadata
    // ========================================================

    if (
        role === "assistant" &&
        metadata
    ) {


        // ----------------------------------------------------
        // Verified Sources
        // ----------------------------------------------------

        if (
            metadata.sources &&
            metadata.sources.length > 0
        ) {

            const sources =
                document.createElement("div");

            sources.className =
                "sources";


            const title =
                document.createElement("div");

            title.className =
                "sources-title";

            title.textContent =
                "Verified 3GPP Sources";

            sources.appendChild(
                title
            );


            metadata.sources.forEach(
                (source) => {

                    const sourceCard =
                        createSourceCard(
                            source
                        );

                    sources.appendChild(
                        sourceCard
                    );
                }
            );


            card.appendChild(
                sources
            );
        }


        // ----------------------------------------------------
        // Confidence + Grounding
        // ----------------------------------------------------

        const metadataBar =
            document.createElement("div");

        metadataBar.className =
            "metadata";


        // ----------------------------------------------------
        // Confidence
        // ----------------------------------------------------

        const confidence =
            document.createElement("span");

        const confidenceValue =
            metadata.confidence || "low";

        confidence.className =
            `badge ${confidenceValue}`;

        confidence.textContent =
            `Confidence: ${confidenceValue}`;

        metadataBar.appendChild(
            confidence
        );


        // ----------------------------------------------------
        // Grounding
        // ----------------------------------------------------

        const grounded =
            document.createElement("span");

        if (metadata.grounded) {

            grounded.className =
                "badge grounded";

            grounded.textContent =
                "Grounded in 3GPP";

        } else {

            grounded.className =
                "badge";

            grounded.textContent =
                "Insufficient evidence";
        }

        metadataBar.appendChild(
            grounded
        );


        card.appendChild(
            metadataBar
        );
    }


    // --------------------------------------------------------
    // Add message to chat
    // --------------------------------------------------------

    wrapper.appendChild(
        card
    );

    chat.appendChild(
        wrapper
    );


    // --------------------------------------------------------
    // Scroll to latest message
    // --------------------------------------------------------

    chat.scrollTop =
        chat.scrollHeight;
}


// ============================================================
// Loading Indicator
// ============================================================

function addLoading() {

    const wrapper =
        document.createElement("div");

    wrapper.className =
        "message assistant";

    wrapper.id =
        "loading-message";


    const card =
        document.createElement("div");

    card.className =
        "message-card";


    const loading =
        document.createElement("div");

    loading.className =
        "loading";


    const spinner =
        document.createElement("div");

    spinner.className =
        "spinner";


    const text =
        document.createElement("span");

    text.textContent =
        "Searching 3GPP standards...";


    loading.appendChild(
        spinner
    );

    loading.appendChild(
        text
    );


    card.appendChild(
        loading
    );


    wrapper.appendChild(
        card
    );


    chat.appendChild(
        wrapper
    );


    chat.scrollTop =
        chat.scrollHeight;
}


// ============================================================
// Remove Loading Indicator
// ============================================================

function removeLoading() {

    const loading =
        document.getElementById(
            "loading-message"
        );

    if (loading) {
        loading.remove();
    }
}


// ============================================================
// Submit Question
// ============================================================

form.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();


        const question =
            questionInput.value.trim();


        if (!question) {
            return;
        }


        // ----------------------------------------------------
        // Show user question
        // ----------------------------------------------------

        addMessage(
            "user",
            question
        );


        questionInput.value = "";

        questionInput.style.height =
            "auto";


        sendButton.disabled =
            true;


        addLoading();


        try {

            // ------------------------------------------------
            // Send question to Flask
            // ------------------------------------------------

            const response =
                await fetch(
                    "/api/ask",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            question: question
                        })
                    }
                );


            const data =
                await response.json();


            removeLoading();


            // ------------------------------------------------
            // API error
            // ------------------------------------------------

            if (
                !response.ok ||
                !data.success
            ) {

                addMessage(
                    "assistant",
                    data.error ||
                    "Something went wrong while processing your question."
                );

                return;
            }


            // ------------------------------------------------
            // Assistant response
            // ------------------------------------------------

            addMessage(
                "assistant",
                data.answer,
                {
                    sources:
                        data.sources || [],

                    confidence:
                        data.confidence || "low",

                    grounded:
                        data.grounded === true
                }
            );


        } catch (error) {

            removeLoading();


            addMessage(
                "assistant",
                "Unable to connect to the Flask server. Please make sure the Flask application is running."
            );


            console.error(
                "API Error:",
                error
            );


        } finally {

            sendButton.disabled =
                false;

            questionInput.focus();
        }
    }
);


// ============================================================
// Auto Grow Textarea
// ============================================================

questionInput.addEventListener(
    "input",
    function () {

        this.style.height =
            "auto";

        this.style.height =
            `${Math.min(
                this.scrollHeight,
                140
            )}px`;
    }
);


// ============================================================
// Enter / Shift + Enter
// ============================================================

questionInput.addEventListener(
    "keydown",
    function (event) {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            form.requestSubmit();
        }
    }
);
const form = document.getElementById("question-form");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send-button");
const chat = document.getElementById("chat");
const welcome = document.getElementById("welcome");


/* ============================================================
   Helpers
============================================================ */

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text ?? "";
    return div.innerHTML;
}


function formatDocumentName(filename) {

    if (!filename) {
        return "Unknown document";
    }

    let name = filename
        .replace(/\.pdf$/i, "")
        .replace(/^ts_/i, "TS ");

    // Convert common filename format:
    // 123501v171500p -> 23.501 Release 17
    const match = name.match(
        /123(\d{3})v(\d{2})(\d{2})(\d{2})p/i
    );

    if (match) {
        const specification = `23.${match[1]}`;
        const release = match[2];

        return `3GPP TS ${specification} • Release ${release}`;
    }

    // Generic formatting fallback
    return name.toUpperCase();
}


function createSourceCard(source) {

    const sourceCard =
        document.createElement("div");

    sourceCard.className =
        "source-card";


    /* --------------------------------------------------------
       Source document name
    -------------------------------------------------------- */

    const sourceName =
        document.createElement("div");

    sourceName.className =
        "source-name";

    sourceName.textContent =
        formatDocumentName(
            source.document
        );


    /* --------------------------------------------------------
       Source metadata
    -------------------------------------------------------- */

    const sourceMeta =
        document.createElement("div");

    sourceMeta.className =
        "source-meta";

    sourceMeta.textContent =
        `Page ${source.page} • Retrieval distance ${source.distance}`;


    sourceCard.appendChild(
        sourceName
    );

    sourceCard.appendChild(
        sourceMeta
    );


    /* --------------------------------------------------------
       Retrieved PDF page image
    -------------------------------------------------------- */

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
            `${formatDocumentName(source.document)} - Page ${source.page}`;

        image.loading =
            "lazy";


        /* ----------------------------------------------------
           Image error handling
        ---------------------------------------------------- */

        image.onerror = function () {

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


    /* --------------------------------------------------------
       Message label
    -------------------------------------------------------- */

    const label =
        document.createElement("div");

    label.className =
        "message-label";

    label.textContent =
        role === "user"
            ? "You"
            : "3GPP Assistant";


    /* --------------------------------------------------------
       Message text
    -------------------------------------------------------- */

    const messageText =
        document.createElement("div");

    messageText.className =
        "message-text";

    messageText.innerHTML =
        escapeHtml(text);


    card.appendChild(
        label
    );

    card.appendChild(
        messageText
    );


    /* ========================================================
       Assistant metadata
    ======================================================== */

    if (
        role === "assistant" &&
        metadata
    ) {


        /* ----------------------------------------------------
           Verified sources
        ---------------------------------------------------- */

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
                "Verified sources";


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


        /* ----------------------------------------------------
           Confidence + grounding
        ---------------------------------------------------- */

        const metadataBar =
            document.createElement("div");

        metadataBar.className =
            "metadata";


        const confidence =
            document.createElement("span");

        confidence.className =
            `badge ${metadata.confidence || "low"}`;

        confidence.textContent =
            `Confidence: ${
                metadata.confidence || "low"
            }`;


        metadataBar.appendChild(
            confidence
        );


        const grounded =
            document.createElement("span");

        grounded.className =
            "badge grounded";


        if (metadata.grounded) {

            grounded.textContent =
                "Grounded";

        } else {

            grounded.textContent =
                "Not grounded";
        }


        metadataBar.appendChild(
            grounded
        );


        /* ----------------------------------------------------
           Retrieval distance
        ---------------------------------------------------- */

        if (
            metadata.retrieval_distance !== null &&
            metadata.retrieval_distance !== undefined
        ) {

            const distance =
                document.createElement("span");

            distance.className =
                "badge grounded";

            distance.textContent =
                `Evidence distance: ${metadata.retrieval_distance}`;


            metadataBar.appendChild(
                distance
            );
        }


        /* ----------------------------------------------------
           Groq invocation status
        ---------------------------------------------------- */

        if (
            metadata.groq_called !== undefined
        ) {

            const groqStatus =
                document.createElement("span");

            groqStatus.className =
                "badge grounded";

            groqStatus.textContent =
                metadata.groq_called
                    ? "LLM invoked"
                    : "LLM not invoked";


            metadataBar.appendChild(
                groqStatus
            );
        }


        card.appendChild(
            metadataBar
        );
    }


    wrapper.appendChild(
        card
    );

    chat.appendChild(
        wrapper
    );


    /* --------------------------------------------------------
       Scroll
    -------------------------------------------------------- */

    chat.scrollTop =
        chat.scrollHeight;
}


/* ============================================================
   Loading indicator
============================================================ */

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
        "Searching 3GPP documents and generating answer...";


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


function removeLoading() {

    const loading =
        document.getElementById(
            "loading-message"
        );

    if (loading) {
        loading.remove();
    }
}


/* ============================================================
   Submit Question
============================================================ */

form.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();


        const question =
            questionInput.value.trim();


        if (!question) {
            return;
        }


        /* ----------------------------------------------------
           Show user question
        ---------------------------------------------------- */

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


            /* ------------------------------------------------
               API error
            ------------------------------------------------ */

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


            /* ------------------------------------------------
               Assistant response
            ------------------------------------------------ */

            addMessage(
                "assistant",
                data.answer,
                {
                    sources:
                        data.sources || [],

                    confidence:
                        data.confidence || "low",

                    grounded:
                        data.grounded === true,

                    retrieval_distance:
                        data.retrieval_distance ?? null,

                    groq_called:
                        data.groq_called
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


/* ============================================================
   Auto-grow textarea
============================================================ */

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


/* ============================================================
   Enter / Shift + Enter
============================================================ */

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
); q
# Google AI Studio setup

1. Sign in at https://aistudio.google.com/apikey using your own Google account.
2. Accept the service terms if you choose to use the service. Create a key in a free-tier project. New accounts may have a default project/key; existing Cloud accounts may need to import a project.
3. Confirm the project is on the **free tier**. Do not enable paid billing or purchase anything for this project. Google may require project verification for free-tier access. Availability and quotas depend on account, region and model.
4. Open the local `.env` file in VS Code. Enter the key after `GEMINI_API_KEY=`; do not include it in chat, screenshots, or Git commits.
5. Keep `LLM_PROVIDER=gemini` and `GEMINI_MODEL=gemini-3.5-flash-lite`. This model's documented standard text input/output currently has a free tier. If unavailable in your account, choose another documented free-tier structured-output model after checking its pricing.
6. Save, then open MoodMix and select **Playlist Generator → llm**. The sidebar reports configuration readiness, which does not guarantee account access or free quota. Settings are read on each backend request, so changes to `.env` do not require a restart; backend code changes do.
7. A live model result shows **Actual planner: llm** or **llm_repaired**. The latter means deterministic code fitted constraints using only the model-proposed tracks. A baseline/fallback result is not evidence of Gemini success. A quota error means wait/check AI Studio; no billing upgrade is required by this application.

Google AI Studio is where the key and project are managed; MoodMix calls the Gemini Developer API. The app cannot determine the project's billing status from a key. A free-tier model can also incur charges if used through a paid-tier project, so retain the free-tier project setting.

Only musical preferences and retrieved track metadata are sent; recordings and credentials are not included in the prompt. Google's free-tier content may be used to improve its products. No audio files are currently uploaded to Google.

The integration uses existing `httpx`, so no additional installation is needed. Automated tests use mock HTTP transport and never call Google. Raw provider response bodies are omitted from user-visible errors.

Sources checked November 11, 2026:

- https://ai.google.dev/gemini-api/docs/api-key
- https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite
- https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite
- https://ai.google.dev/gemini-api/docs/structured-output
- https://ai.google.dev/api/generate-content

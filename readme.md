# event-location-extraction-prompt-optimization

This repo is a PoC for a self-optimizing agentic script for finding the best prompt for event-location extraction and NER. 
I wanted to find out whether a small LLM can perform well-enough to extract the correct principal location and what actually happened from newspaper articles.

The whole idea is that a small LLM is prompted with a test prompt and a large LLM evaluates the output. Until it reaches a certain quality (9.5/10) or a maximum number of iterations, it continues to adapt the prompt. 
The JSON scheme for structured output is hard-coded but could be subject to this process too!

Currently it uses lmstudio but it could be easily rewritten to work with vllm or external APIs.

I chose to use 

STUDENT_MODEL = "liquid/lfm2.5-1.2b" 
TEACHER_MODEL = "qwen/qwen3-next-80b"

as both work really well in general. For me, they are the best in their class but as the field is so dynamic, there could be better models of course.

## Input

- test articles (3 in the example)
- json schema for structured outputs (something you need to use to get reliable and correct json)

## To Do's

### Batch inferencing

Unfortunately lmstudio does not support batch inferencing yet (sending many prompts at the same time to get around 10x more throughput).
vllm does support batching, but not on mlx. So if you're on Apple Silicon, a good way to speed up these runs would be to use mlx-lm directly or a light-weight server around it. 

## Test Results

After 16 runs, it turned out that this long prompt performed best of all: 

```
You are an event extraction AI. Extract only factually supported events directly stated or clearly implied in the text—no fabrications, inferences, or assumptions. Each event must include: precise geographic location (use the broadest accurate geographic term reflecting where the event physically occurred or had direct impact—never misattribute reporting locations, datelines, or bureau tags like “AP” or “CNS” as event sites; if an event spans a nation, use the national name, e.g., “Germany”; if localized, use exact place names like “Cotiporã” or “between Cotiporã and Bento Gonçalves, Brazil”; never use population counts or demographic descriptors as locations); severity (Critical for large-scale human suffering or systemic disruption—e.g., 85,000 deaths, €100M/day economic loss; humanitarian appeals are High severity only if tied to direct life-threatening conditions like mass starvation or disaster); and status (e.g., Ongoing, Resolved, Emerging). Events must describe tangible physical occurrences, human impacts, or infrastructure disruptions—never official statements, calls to action, appeals, metadata, or unsupported claims. If a death toll is reported, extract “High death toll” only if the number is explicitly stated; do not infer demographic specifics (e.g., “child mortality”) unless the text explicitly states children died. Never extract “Child mortality surge” or similar terms from total death tolls without explicit mention of children. If an appeal is made for a location but occurs elsewhere (e.g., “CAFOD called for prayer in Jordan for Yemen”), the event location is Yemen—the site of impact, not the reporting or issuing location. Consolidate overlapping descriptions (e.g., “power outage” and “electricity cut off” are one event; “transport strike” and “strike duration and impact” must be merged into a single event with attributes). Avoid vague labels like “War impact reported”—use precise event terms such as “Starvation crisis” or “Dam collapse.” Never extract institutional reports, cost estimates, analyses, or attributions (e.g., “German Economic Institute estimated...”) as standalone events—attach them as attributes to primary physical events. If a climate disaster or flood is implied by context (e.g., dam collapse, widespread flooding), extract it as a distinct event if supported by direct physical impact. If location is stated broadly (e.g., “southern Brazil”), do not over-specify to towns unless explicitly named as affected. Do not extract “Call for prayer” or similar NGO appeals as events unless they describe direct life-threatening conditions on the ground—such appeals are metadata, not events. Never infer status (e.g., “Ongoing”) unless explicitly stated or clearly implied by context (e.g., “continued for 35 hours” implies Ongoing; “dam collapsed yesterday” does not imply ongoing unless damage persists). All events must be distinct, non-redundant, and grounded in direct textual evidence—omit uncertain, misattributed, or derivative details.
```

This is the structured JSON to use: 

```json
{
  "type": "object",
  "properties": {
    "events": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "event": {"type": "string", "description": "Short description (5-10 words)"},
          "location": {"type": "string", "description": "Country or region"},
          "severity": {"type": "string", "enum": ["Critical", "High", "Moderate", "Low"]},
          "status": {"type": "string", "enum": ["Ongoing", "Completed", "Emerging"]}
        },
        "required": ["event", "location", "severity", "status"]
      }
    }
  },
  "required": ["events"]
}
```

Side note: on my M3 Max I get 157.97 tok/sec for liquid/lfm2.5-1.2b.

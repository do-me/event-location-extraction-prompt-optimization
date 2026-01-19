# event-location-extraction-prompt-optimization

This repo is a PoC for a self-optimizing agentic script for finding the best prompt for event-location extraction and NER. 
I wanted to find out whether a small LLM can perform well-enough to extract the correct principal location and what actually happened from newspaper articles.

The whole idea is that a small LLM is prompted with a test prompt and a large LLM evaluates the output. Until it reaches a certain quality (9.5/10) or a maximum number of iterations, it continues to adapt the prompt. 
The JSON scheme for structured output is hard-coded but could be subject to this process too!

Currently it uses either lmstudio or mlx-lm but it could be easily rewritten to work with vllm or external APIs. It supports batch mode for mlx-lm but note that some modern models like Mamba/SSM don't support it.

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

This is the basic schema (structured JSON) to use: 

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

This is an optimized schema (structured JSON) to use: 

```json
{
  "type": "object",
  "properties": {
    "events": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "event": {
            "type": "string",
            "description": "Exact phrasing or close paraphrase of the factual event as stated in the text (5-20 words). Must reflect the original wording or meaning without semantic distortion (e.g., 'pressed for' not 'urged'). Events must describe tangible physical occurrences: disasters, infrastructure failures, human casualties, or direct impacts. Do not extract official statements, appeals, attributions, estimates, demographic inferences (e.g., 'child mortality'), or vague labels like 'war impact.' If an event describes an effect (e.g., 'transport halted'), ensure it is grounded in a direct physical cause mentioned in the text. Avoid figurative language unless it directly corresponds to a measurable physical outcome (e.g., 'ground to a halt' is acceptable if supported by evidence of total disruption)."
          },
          "location": {
            "type": "string",
            "description": "Precise geographic location where the event occurred. Use nation if national scope; exact city, region, or site if local. Never use media outlets, organizations, NGOs, reporting sources, or speaker locations as locations. If an appeal or statement is made about a location (e.g., 'CAFOD calls for prayer for Yemen'), the location must be the geographic area affected by the event being described, not the speaker's location. If no tangible physical event occurs in a location, do not extract an event. If the geographic scope is not explicitly stated (e.g., '300,000 residents affected' without location details), do not assign a location beyond the broadest confirmed area (e.g., 'southern Brazil' if the dam is there, but not 'near the dam site' unless confirmed)."
          },
          "severity": {
            "type": "string",
            "enum": [
              "Critical",
              "High",
              "Moderate"
            ],
            "description": "Critical: large-scale death/injury (e.g., death toll >=10), widespread infrastructure collapse, or systemic humanitarian crisis with explicit quantitative or unambiguous qualitative evidence (e.g., 'immense suffering' only if explicitly tied to documented child deaths, mass displacement, or similar concrete physical harm). High: direct life-threatening conditions with explicit quantitative evidence (e.g., starvation affecting X people, acute medical crisis with quantified impact). Moderate: significant disruption without immediate life threat. Severity must be inferable only from explicit quantitative data or unambiguous qualitative indicators directly stated in the text—never inferred from tone, context, attributions, or advocacy actions (e.g., 'calls for prayer' are not severity-worthy events). 'Immense suffering' alone is insufficient for severity classification; it must be explicitly paired with a measurable physical outcome in the same sentence or adjacent clause."
          },
          "status": {
            "type": "string",
            "enum": [
              "Ongoing",
              "Completed",
              "Emerging"
            ],
            "description": "Ongoing: event is actively continuing with explicit temporal indicators (e.g., 'still ongoing', 'continues', 'now in its 35th hour'). Completed: event has definitively ended with explicit end markers (e.g., 'lasted 35 hours', 'ended at 8 PM'). Emerging: event is developing but not yet fully manifest, with explicit indicators (e.g., 'rising', 'escalating', 'expected to worsen'). Status must be explicitly stated or logically deducible from clear temporal or directional language in the text—no speculation, inference, or assumption allowed. If no temporal indicator is present, do not assign status. Duration alone (e.g., '35 hours') does not imply 'Ongoing' unless the event is confirmed to be still active at the time of reporting."
          },
          "evidence": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "minItems": 1,
            "description": "One or more direct quotes or near-verbatim phrases from the text that unambiguously support the event. Each evidence item must be a verbatim or minimally paraphrased excerpt that directly corresponds to the event and location. Evidence must not include attributions (e.g., 'according to X', 'X reported') unless the attribution is part of the factual event itself (e.g., 'Deutsche Bahn noted that only 20% of long-distance trains would run'). Subjective statements (e.g., 'We see immense suffering') are permitted as evidence ONLY if they are explicitly tied to a measurable physical event with direct quantitative support in the same sentence or adjacent clause (e.g., 'We see immense suffering due to 200 civilian deaths'). All evidence must be directly extractable from the text without inference, paraphrasing, or contextual embellishment. Do not omit direct factual details from the text (e.g., death tolls, wave heights, time of onset) that substantiate the event. Do not include standalone subjective statements that lack direct quantitative or physical correlation to the event."
          }
        },
        "required": [
          "event",
          "location",
          "severity",
          "status",
          "evidence"
        ]
      }
    }
  },
  "required": [
    "events"
  ]
}
```

Side note: on my M3 Max I get 157.97 tok/sec for liquid/lfm2.5-1.2b.

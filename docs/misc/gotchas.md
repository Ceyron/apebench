# Gotchas and Sharp Edges

- When subclassing from `BaseScenario` and overriding some of the attributes
    those need to be typed similarly to the base class.
- Configuration strings: many configurations (like network architecture, initial
  condition distribution, optimization config, learning methodology, etc.) are
  set up in terms of string which contains the configuration entries separated
  by a semi-colon `";"`.
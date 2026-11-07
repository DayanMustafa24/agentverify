# Failure analysis (full system, eval set)

4-way errors: 46/150

### Success: val-100
- Caption: Robert Ethan Saylor is seen here in this family photo taken during a summer festival in 2012
- Gold: real | Predicted: real (confidence 0.85)
- text agent: flags real (0.80). The caption makes a mundane, verifiable biographical claim about an individual in a family photo without using sensationalism or implausible details.
- image agent: flags real (0.85). The image displays natural photographic qualities, consistent lighting, and realistic textures without obvious signs of AI generation or digital tampering.
- evidence agent: flags real (0.90). [insufficient] The retrieved evidence mentions Ethan Saylor and family photographs, but does not provide specific details to confirm whether the particular photo described in the caption was taken during a summer festival in 2012.
    - evidence: en.wikipedia.org | List of people from Illinois
    - evidence: ethansayloralliance.org | Ethan’s Story - Ethan Saylor Alliance
    - evidence: facebook.com | Ethan Saylor - Facebook
- Decision explanation: Both the text and image agents confirm the content is natural, realistic, and lacks any indicators of fabrication or manipulation, supported by consistent biographical context.
- Cost: 5 LLM calls, 2662 tokens

### Success: val-701
- Caption: Andy Murray returns the ball to YenHsun Lu he won comfortably also in straight sets
- Gold: mismatch | Predicted: mismatch (confidence 0.95)
- text agent: flags real (0.80). The caption describes a standard tennis match outcome between Andy Murray and Yen-Hsun Lu, containing no sensationalism, implausible details, or absolute claims.
- image agent: flags mismatch (0.95). The image shows Andy Murray holding a trophy and celebrating with a crowd and flag, whereas the caption describes him returning the ball during a match against Yen-Hsun Lu.
- evidence agent: flags real (1.00). [supports] Evidence [5] and [6] confirm that Andy Murray defeated Yen-Hsun Lu in straight sets.
    - evidence: en.wikipedia.org | Lu Yen-hsun
    - evidence: en.wikipedia.org | Andy Murray
    - evidence: en.wikipedia.org | Andy Murray career statistics
- Decision explanation: While the textual claim that Andy Murray defeated Yen-Hsun Lu in straight sets is factually correct, the image depicts a post-match trophy celebration rather than the action described in the caption.
- Cost: 5 LLM calls, 2580 tokens

### Failure: val-700
- Caption: The final emails were released a day before Hillary Clinton competes in 11 Democratic primary contests on Super Tuesday
- Gold: mismatch | Predicted: real (confidence 0.95)
- text agent: flags real (0.85). The caption refers to a well-documented news event regarding the release of Hillary Clinton's emails leading up to Super Tuesday primaries, without containing sensationalism or implausible claims.
- image agent: flags real (0.95). The image shows a genuine photograph of Hillary Clinton speaking at a campaign event, perfectly matching the political context of the caption.
- evidence agent: flags real (0.95). [supports] Evidence [2] confirms the final batch of Hillary Clinton's emails was released on February 29, 2016, which was the day before Super Tuesday (March 1, 2016) when she competed in multiple Democratic primary contests.
    - evidence: en.wikipedia.org | Hillary Clinton 2008 presidential campaign
    - evidence: en.wikipedia.org | Hillary Clinton 2016 presidential campaign
    - evidence: en.wikipedia.org | 2016 Democratic Party presidential primaries
- Decision explanation: All specialist agents independently verified the caption, image, and historical facts, confirming that the release of Hillary Clinton's final emails occurred the day before the Super Tuesday primaries in 2016.
- Cost: 5 LLM calls, 2640 tokens

### Failure: val-373
- Caption: Jack Reacher includes a signature horse chase sequence.
- Gold: text_false | Predicted: image_false (confidence 0.95)
- text agent: flags text_false (0.90). Jack Reacher stories (both the books and film/TV adaptations) famously center on modern, gritty investigations and a nomadic protagonist who typically travels on foot or by bus, making a signature horse chase sequence implausible.
- image agent: flags image_false (0.99). The image exhibits classic AI-generation artifacts, such as garbled, nonsensical text on the storefront sign ('XBERIFF SIUICIE') and unnatural blending in the architectural details and textures.
- evidence agent: flags real (1.00). [insufficient] None of the provided evidence items mention a horse chase sequence in Jack Reacher.
    - evidence: en.wikipedia.org | Slow Horses
    - evidence: en.wikipedia.org | The Shining (film)
    - evidence: en.wikipedia.org | The Mummy (1999 film)
- Decision explanation: The image contains severe AI-generation artifacts such as garbled text, and the caption falsely claims a signature horse chase in Jack Reacher.
- Cost: 5 LLM calls, 2587 tokens

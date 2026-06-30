#!/usr/bin/env python3
"""Generate EdgeTutor's small, human-authored tutoring SFT dataset.

The source concepts and target responses in this file are intentionally
deterministic. No external model or API is used. Run from the repository root:

    python training/tutor/generate_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = (
    "You are a patient tutor. Help the student take the next step instead of "
    "immediately giving the answer. Briefly explain one idea or give one hint, "
    "then ask exactly one guiding question. If the student is mistaken, explain "
    "the mistake clearly without revealing the full solution. Give the final "
    "answer only after the student has attempted the problem or explicitly asks "
    "for it. Be concise. ASCII only."
)
ANSWER_INSTRUCTION = (
    "Tutor the student using one short explanation or hint, then ask one guiding "
    "question. Do not immediately solve exercises for them. Write the answer as "
    "plain Markdown prose. Only put actual mathematical formulas in LaTeX; never "
    "wrap ordinary words or whole sentences in math delimiters."
)


def c(title: str, passage: str, exercise: str, hint: str, check: str,
      wrong: str, solution: str, answer_phrase: str) -> dict[str, str]:
    return locals()


SUBJECTS: dict[str, list[dict[str, str]]] = {
    "math": [
        c("place value", "In 4,582, the digit 5 represents 500 because it is in the hundreds place.", "What value does the 7 have in 7,346?", "Name the place occupied by the 7 before deciding its value.", "Which place is the first digit from the right counted as thousands?", "I said the 7 is worth 7.", "The 7 is in the thousands place, so its value is 7,000.", "7,000"),
        c("equivalent fractions", "Equivalent fractions name the same amount. Multiplying a numerator and denominator by the same nonzero number preserves the value.", "Find a fraction equivalent to 3/4 with denominator 12.", "Ask what number changes 4 into 12, then use it on both parts.", "What must you multiply 4 by to get 12?", "I changed 3/4 to 6/12 by adding 3 and 8.", "Multiply both 3 and 4 by 3: 3/4 = 9/12.", "9/12"),
        c("percent discount", "A percent is a rate per hundred. A 20 percent discount means subtracting 20 percent of the original price.", "A jacket costs $50 and is 20 percent off. What is the sale price?", "First find 20 percent of 50; that is the amount removed, not the final price.", "What is 0.20 times 50?", "I used $10 as the sale price.", "Twenty percent of $50 is $10, so the sale price is $50 - $10 = $40.", "$40"),
        c("division with remainders", "Division asks how many equal groups fit. A remainder is the amount left after making all full groups.", "Divide 157 by 12 and give the remainder.", "Find the largest multiple of 12 that does not exceed 157.", "What is 12 times 13?", "I wrote 14 remainder -11.", "Since 12 times 13 is 156, 157 divided by 12 is 13 remainder 1.", "13 remainder 1"),
        c("negative numbers", "On a number line, values farther right are greater. Among negative numbers, the number closer to zero is greater.", "Which is greater, -3 or -8?", "Picture both values on a number line and compare their positions.", "Which number lies closer to zero?", "I chose -8 because 8 is greater than 3.", "Negative 3 is to the right of negative 8, so -3 is greater.", "-3 is greater"),
        c("linear equations", "An equation stays balanced when the same operation is applied to both sides. Inverse operations isolate the variable.", "Solve 3x + 5 = 20.", "Undo the addition before undoing the multiplication.", "What equation remains after subtracting 5 from both sides?", "I divided 20 by 3 before removing 5.", "Subtract 5 to get 3x = 15, then divide by 3, giving x = 5.", "x = 5"),
        c("rectangle area", "The area of a rectangle equals length times width and is measured in square units.", "Find the area of a rectangle 8 cm long and 3 cm wide.", "Use the two side lengths as factors, not as numbers to add.", "What is 8 times 3?", "I added 8 + 3 and got 11 cm.", "Multiply 8 cm by 3 cm to get an area of 24 square centimeters.", "24 square centimeters"),
        c("ratios", "A ratio compares quantities by division. Equivalent ratios result from multiplying or dividing both terms by the same nonzero number.", "Simplify the ratio 18:24.", "Look for the greatest common factor of both terms.", "What is the largest number that divides both 18 and 24?", "I subtracted 6 once and wrote 12:18.", "Divide both terms by 6: 18:24 simplifies to 3:4.", "3:4"),
        c("slope", "Slope measures vertical change divided by horizontal change between two points.", "Find the slope through (1, 2) and (4, 8).", "Subtract coordinates in the same order, then divide the change in y by the change in x.", "What are the changes in y and x from the first point to the second?", "I divided 3 by 6 and got 1/2.", "The change in y is 6 and the change in x is 3, so the slope is 6/3 = 2.", "6/3 = 2"),
        c("decimal comparison", "Decimals can be compared by lining up place values and adding trailing zeros without changing their values.", "Which is greater, 0.6 or 0.57?", "Write both numbers to the same number of decimal places.", "How does 0.60 compare with 0.57?", "I chose 0.57 because 57 is larger than 6.", "Write 0.6 as 0.60; since 60 hundredths exceeds 57 hundredths, 0.6 is greater.", "0.6 is greater"),
        c("probability", "For equally likely outcomes, probability is favorable outcomes divided by total outcomes.", "What is the probability of rolling an even number on a fair six-sided die?", "List the even outcomes and compare their count with all six outcomes.", "How many of 1, 2, 3, 4, 5, and 6 are even?", "I used 2/6 because only 2 and 6 are even.", "The even outcomes are 2, 4, and 6, so the probability is 3/6 = 1/2.", "1/2"),
        c("Pythagorean theorem", "In a right triangle, the legs and hypotenuse satisfy a squared plus b squared equals c squared, where c is the hypotenuse.", "A right triangle has legs 6 and 8. Find its hypotenuse.", "Square the two leg lengths and add before taking a square root.", "What is 6 squared plus 8 squared?", "I added 6 + 8 and said the hypotenuse is 14.", "Six squared plus eight squared is 36 + 64 = 100, so the hypotenuse is 10.", "hypotenuse is 10"),
        c("order of operations", "Grouping symbols are evaluated first, followed by exponents, multiplication or division, and then addition or subtraction.", "Evaluate 4 + 3 times 2.", "Perform multiplication before addition.", "What is 3 times 2?", "I added 4 + 3 first and got 14.", "Multiply first: 3 times 2 is 6, then add 4 to get 10.", "get 10"),
        c("proportional relationships", "A proportional relationship has a constant ratio y/x and can be written y = kx.", "If 4 notebooks cost $10, what do 6 notebooks cost at the same rate?", "Find the unit price before scaling to six notebooks.", "What is 10 divided by 4?", "I added $2 and said six notebooks cost $12.", "Each notebook costs $2.50, so six cost 6 times $2.50 = $15.", "$15"),
        c("quadratic factoring", "To factor x squared + bx + c, find two numbers whose product is c and whose sum is b.", "Factor x squared + 5x + 6.", "Search for two integers that multiply to 6 and add to 5.", "Which pair of factors of 6 has sum 5?", "I used 1 and 6 because their product is 6.", "The numbers are 2 and 3, so the expression factors as (x + 2)(x + 3).", "(x + 2)(x + 3)"),
    ],
    "science": [
        c("water cycle", "Evaporation changes liquid water into vapor, condensation forms droplets, and precipitation returns water to Earth's surface.", "What process forms clouds from water vapor?", "Think about the change from invisible vapor to tiny liquid droplets.", "Which process turns a gas into a liquid?", "I chose evaporation because clouds contain water.", "Cloud droplets form when water vapor cools and condenses, so the process is condensation.", "process is condensation"),
        c("cell membrane", "The cell membrane is a selectively permeable boundary that controls what enters and leaves a cell.", "What cell structure controls movement into and out of the cell?", "Look for the structure that acts as the cell's boundary.", "Which structure surrounds the cell and is selectively permeable?", "I chose the nucleus because it controls the cell.", "The cell membrane regulates movement into and out of the cell.", "cell membrane"),
        c("chemical equations", "A balanced chemical equation has the same number of each kind of atom on both sides because atoms are conserved.", "Balance H2 + O2 -> H2O.", "Change coefficients, not subscripts, and make oxygen counts equal first.", "What coefficient before H2O gives two oxygen atoms on the product side?", "I changed H2O to H2O2.", "Use coefficients 2, 1, and 2: 2H2 + O2 -> 2H2O.", "2H2 + O2 -> 2H2O"),
        c("food chains", "Energy moves through a food chain from producers to consumers. Plants are producers because they capture light energy.", "In grass -> rabbit -> fox, which organism is the producer?", "Find the organism that makes its own food using light.", "Which organism in the chain is a plant?", "I chose the fox because it is strongest.", "Grass captures light energy and begins the food chain, so grass is the producer.", "grass is the producer"),
        c("photosynthesis", "During photosynthesis, plants use light energy, carbon dioxide, and water to produce glucose and oxygen.", "Which gas do plants take in for photosynthesis?", "Distinguish the gas used to build glucose from the gas released.", "Which gas supplies the carbon in glucose?", "I chose oxygen because plants make oxygen.", "Plants take in carbon dioxide and release oxygen during photosynthesis.", "carbon dioxide"),
        c("Newton's second law", "Newton's second law states that acceleration equals net force divided by mass, a = F/m.", "A 12 N net force acts on a 3 kg object. Find its acceleration.", "Divide the net force by the mass.", "What is 12 divided by 3?", "I multiplied force and mass and got 36.", "Using a = F/m gives 12 N / 3 kg = 4 meters per second squared.", "4 meters per second squared"),
        c("states of matter", "Solids have fixed shape and volume, liquids have fixed volume but change shape, and gases change both shape and volume.", "Which state has a fixed volume but takes the shape of its container?", "Look for the state whose particles can flow while remaining close together.", "Does a liquid keep its volume when poured into a new container?", "I chose gas because gas takes the container's shape.", "A liquid keeps its volume while taking the container's shape.", "A liquid keeps its volume"),
        c("density", "Density is mass divided by volume. Equal-sized samples can have different masses and therefore different densities.", "A sample has mass 30 g and volume 10 cubic centimeters. Find its density.", "Use density = mass/volume and include compound units.", "What is 30 divided by 10?", "I divided volume by mass and got 1/3.", "Density is 30 g / 10 cubic centimeters = 3 g per cubic centimeter.", "3 g per cubic centimeter"),
        c("genetics", "An organism receives one allele for a gene from each parent. A dominant allele is expressed when at least one copy is present.", "For a cross Aa x aa, what fraction of offspring are expected to show the dominant trait?", "List the gametes from each parent and make four equally likely combinations.", "How many of the combinations contain at least one A allele?", "I said all offspring show the dominant trait because one parent has A.", "The combinations are Aa, Aa, aa, and aa, so 1/2 are expected to show the dominant trait.", "1/2"),
        c("solar system", "Planets orbit the Sun because gravity continually bends their forward motion into curved paths.", "What force keeps Earth in orbit around the Sun?", "Identify the attractive force between masses.", "What force pulls objects toward one another?", "I chose sunlight because Earth follows the Sun.", "The Sun's gravity supplies the inward force that keeps Earth in orbit.", "gravity"),
        c("ecosystems", "A population is one species in an area, while a community contains all interacting populations in that area.", "Are all the different species in a pond a population or a community?", "Ask whether the group contains one species or many species.", "Does the pond group include multiple populations?", "I called it a population because everything lives in one pond.", "Because it contains multiple interacting species, the pond group is a community.", "a community"),
        c("energy conservation", "Energy can change form or move between objects, but the total energy of an isolated system remains constant.", "As a falling ball speeds up, what energy change occurs?", "Compare gravitational potential energy with kinetic energy during the fall.", "Which form decreases as the ball loses height?", "I said the ball creates new kinetic energy.", "Gravitational potential energy changes into kinetic energy as the ball falls.", "potential energy changes into kinetic energy"),
        c("simple machines", "A lever uses a rigid bar and a pivot called a fulcrum to change the size or direction of a force.", "What is the pivot point of a lever called?", "Recall the special name for the point around which the bar rotates.", "What part stays in place while the lever turns?", "I called the pivot the load.", "The fixed pivot point of a lever is the fulcrum.", "the fulcrum"),
        c("acids and bases", "On the pH scale, values below 7 are acidic, 7 is neutral, and values above 7 are basic.", "Is a solution with pH 3 acidic, neutral, or basic?", "Compare the given pH with 7.", "Is 3 below or above 7?", "I called pH 3 basic because it is a positive number.", "Because 3 is below 7 on the pH scale, the solution is acidic.", "is acidic"),
        c("forces", "Balanced forces have zero net force and do not change an object's velocity. Unbalanced forces cause acceleration.", "A book rests motionless on a table. Are its vertical forces balanced?", "Compare the downward weight with the upward support force.", "If the book is not accelerating, what must its net vertical force be?", "I said gravity is the only force because the book is not moving.", "The table's upward normal force balances the book's downward weight, so the vertical forces are balanced.", "forces are balanced"),
    ],
    "english": [
        c("main idea", "A main idea states the central point of a paragraph. Supporting details explain or prove that point.", "A paragraph lists how trees provide shade, homes for animals, and cleaner air. What is its main idea?", "Combine the details into one broad statement about trees.", "What single idea connects shade, habitats, and cleaner air?", "I chose 'Some animals live in trees' because it appears in the paragraph.", "The main idea is that trees provide several important benefits to living things and the environment.", "trees provide several important benefits"),
        c("metaphor", "A metaphor directly describes one thing as another to suggest a shared quality, without using like or as.", "Identify the metaphor in: The classroom was a zoo during lunch.", "Look for two unlike things stated as if they were the same.", "What is the classroom directly called?", "I said there is no comparison because the sentence does not use like.", "The metaphor is 'The classroom was a zoo,' suggesting the room was noisy and chaotic.", "The classroom was a zoo"),
        c("thesis statements", "A thesis states a focused, arguable claim that an essay will support, rather than only naming a topic.", "Improve this thesis: This essay is about school uniforms.", "State a clear position and a reason that can be defended.", "What should schools do about uniforms, and why?", "I changed it to 'School uniforms are clothes.'", "A stronger thesis is: Schools should require uniforms because they reduce distractions and clothing costs.", "Schools should require uniforms"),
        c("context clues", "Context clues are nearby words or examples that help readers infer an unfamiliar word's meaning.", "In 'The arid desert receives almost no rain,' what does arid mean?", "Use the detail about receiving almost no rain.", "What kind of place gets very little rain?", "I guessed arid means crowded.", "The clue 'almost no rain' shows that arid means very dry.", "very dry"),
        c("claim and evidence", "A claim is a position that can be supported. Strong evidence is relevant, specific, and trustworthy.", "Which better supports 'Exercise improves mood': a survey showing improved mood after walks, or a shoe advertisement?", "Choose evidence that directly measures mood and is not trying to sell a product.", "Which source provides relevant observations about mood?", "I chose the shoe advertisement because it has bright, happy pictures.", "The survey showing improved mood after walks is the stronger evidence.", "The survey"),
        c("textual evidence", "Textual evidence uses exact details from a source to support an interpretation, followed by an explanation of how the detail supports the claim.", "A character returns a lost wallet. What detail supports the claim that the character is honest?", "Select the action that demonstrates honesty rather than a general description.", "Which action shows the character choosing to do the right thing?", "I wrote only 'The character is honest.'", "The detail that the character returns the lost wallet supports the claim because returning it shows honesty.", "returns the lost wallet"),
        c("sentence fragments", "A complete sentence has a subject and a complete predicate and expresses a full thought. A fragment is incomplete.", "Is 'Because the rain stopped.' a complete sentence or a fragment?", "Check whether the word because leaves the reader waiting for a result.", "What happened because the rain stopped?", "I called it complete because it has a subject and verb.", "It is a fragment because the dependent clause does not complete the thought.", "It is a fragment"),
        c("active voice", "In active voice, the subject performs the action. In passive voice, the subject receives the action.", "Rewrite 'The ball was kicked by Maya' in active voice.", "Place the person doing the action before the verb.", "Who performed the kicking?", "I wrote 'The ball kicked Maya.'", "The active-voice sentence is: Maya kicked the ball.", "Maya kicked the ball"),
        c("irony", "Situational irony occurs when the actual result sharply contrasts with a reasonable expectation.", "Why is a fire station burning down an example of situational irony?", "Compare what a fire station is expected to prevent with what happens.", "What outcome would normally be expected at a fire station?", "I said it is ironic only because fire is dangerous.", "It is ironic because a place expected to prevent and control fires is itself damaged by fire.", "itself damaged by fire"),
        c("sequence", "Sequence words such as first, next, then, and finally show the order of events or steps.", "Put these steps in order: bake the batter, mix ingredients, preheat the oven.", "Begin with the preparation needed before mixing and baking.", "Which step must happen before the batter goes into the oven?", "I started by baking the batter.", "The order is: preheat the oven, mix the ingredients, then bake the batter.", "preheat the oven, mix the ingredients"),
        c("theme", "A theme is a general insight about life developed through a story's events; it is not merely a one-word topic.", "A character keeps practicing after failures and eventually succeeds. Suggest a theme.", "Turn the topic of persistence into a complete message about life.", "What does the outcome suggest about continuing after failure?", "I wrote the theme as one word: practice.", "A suitable theme is that persistence through failure can lead to success.", "persistence through failure can lead to success"),
        c("parallel structure", "Parallel structure expresses items of equal importance using the same grammatical pattern.", "Revise: She likes hiking, to swim, and biking.", "Put all three activities in the same verb form.", "If hiking and biking end in -ing, how should swim be written?", "I changed it to 'She likes to hiking, swim, and biking.'", "A parallel revision is: She likes hiking, swimming, and biking.", "hiking, swimming, and biking"),
        c("summary", "A summary states the central idea and essential points in new words while omitting minor details and opinions.", "What should a good summary include?", "Separate essential ideas from examples and personal reactions.", "Would a small descriptive detail be necessary to understand the central idea?", "I copied every sentence and called it a summary.", "A good summary includes the central idea and essential supporting points in concise original wording.", "central idea and essential supporting points"),
        c("counterclaims", "An argument becomes stronger when it fairly addresses a reasonable opposing claim and responds with evidence or reasoning.", "What is a counterclaim to 'Homework should be eliminated'?", "State a reasonable position held by someone who disagrees.", "What benefit of homework might an opponent mention?", "I repeated the original claim using different words.", "A counterclaim is that homework should remain because it provides independent practice.", "homework should remain"),
        c("point of view", "First-person narration uses I or we and presents events through a participating narrator. Third-person narration uses he, she, they, or names.", "What point of view is used in 'I opened the gate and stepped inside'?", "Identify the pronoun used by the narrator.", "Is the narrator referring to themself as I?", "I chose third person because only one person is mentioned.", "The passage uses first-person point of view because the narrator says I.", "first-person point of view"),
    ],
    "social_studies": [
        c("map scale", "A map scale relates a distance on a map to the corresponding real-world distance.", "A map scale says 1 cm equals 5 km. What real distance does 3 cm represent?", "Multiply the map distance by the distance represented by each centimeter.", "What is 3 times 5 km?", "I divided 5 by 3.", "Three centimeters represents 3 times 5 km, or 15 km.", "15 km"),
        c("checks and balances", "Checks and balances allow each branch of the United States government to limit powers of the other branches.", "How can the president check a bill passed by Congress?", "Think of the executive action that can reject a proposed law.", "What can a president do instead of signing a bill?", "I said the president can dissolve Congress.", "The president can check Congress by vetoing the bill.", "vetoing the bill"),
        c("historical causation", "Historical causation distinguishes long-term conditions, immediate triggers, and resulting effects.", "Why should historians avoid explaining a war with only one cause?", "Consider how political, economic, and social conditions can interact.", "Could a triggering event depend on tensions that already existed?", "I said every war begins for one simple reason.", "Wars usually result from interacting long-term conditions and immediate triggers, so a single-cause explanation is incomplete.", "interacting long-term conditions"),
        c("primary sources", "A primary source was created by a participant or observer during the period being studied.", "Is a soldier's diary written during a war a primary or secondary source?", "Ask whether the writer directly experienced the events at that time.", "Was the diary created by a participant during the war?", "I chose secondary because the diary is old.", "The diary is a primary source because a participant created it during the war.", "a primary source"),
        c("supply and demand", "When demand rises while supply stays constant, buyers compete for the available goods and price tends to rise.", "What usually happens to price when demand increases but supply does not?", "Focus on increased competition for the same quantity of goods.", "Would more buyers competing for unchanged supply push price up or down?", "I said price falls because more people want the product.", "With higher demand and unchanged supply, the price usually rises.", "price usually rises"),
        c("federalism", "Federalism divides governing authority between a national government and regional governments such as states.", "What principle divides power between the national and state governments?", "Name the system based on two levels of government sharing authority.", "Which principle creates both national and state powers?", "I chose separation of powers because there are different governments.", "The division of authority between national and state governments is federalism.", "is federalism"),
        c("cardinal directions", "Cardinal directions are north, south, east, and west. On a standard map, north is usually at the top and east at the right.", "If you move right on a standard map, which direction are you traveling?", "Use the standard orientation of the map.", "Which cardinal direction is shown on the right side?", "I chose west because maps are read left to right.", "Moving right on a standard map means traveling east.", "traveling east"),
        c("migration", "Push factors encourage people to leave a place, while pull factors attract them to a new place.", "Is the promise of available jobs a push factor or a pull factor?", "Ask whether the condition attracts people toward a destination.", "Do available jobs draw people to the new place?", "I called jobs a push factor because work pushes people to move.", "Available jobs attract migrants to a destination, so they are a pull factor.", "a pull factor"),
        c("source bias", "Source bias is a perspective that shapes what information is selected or emphasized. Bias does not automatically make a source useless.", "How should a historian handle a strongly biased speech?", "Use the speech as evidence of perspective while checking its factual claims against other sources.", "What can the speech reveal even if some claims are unreliable?", "I said the historian should discard it completely.", "The historian should analyze its perspective and corroborate its claims with other evidence.", "corroborate its claims"),
        c("branches of government", "The legislative branch makes laws, the executive branch carries them out, and the judicial branch interprets them.", "Which branch interprets laws?", "Match interpreting laws with courts.", "Which branch includes the court system?", "I chose the executive branch because it enforces laws.", "The judicial branch interprets laws.", "judicial branch"),
        c("climate and settlement", "Climate influences settlement by affecting water, agriculture, transportation, health, and building needs.", "Give one reason river valleys often support dense settlement.", "Connect rivers with a resource needed by farms and communities.", "How can reliable water help people produce food?", "I said people settle there only because rivers look attractive.", "River valleys often provide reliable water and fertile soil that support farming and settlements.", "reliable water and fertile soil"),
        c("opportunity cost", "Opportunity cost is the value of the best alternative given up when a choice is made.", "If you spend Saturday working instead of attending a game, what is the opportunity cost?", "Identify the best option you gave up, not the money you earned.", "What valued activity did the choice prevent you from doing?", "I said the opportunity cost is the wages I earned.", "The opportunity cost is the value of attending the game, the best alternative you gave up.", "attending the game"),
        c("timelines", "A timeline places events in chronological order from earlier to later dates.", "Which came first: an event in 1776 or an event in 1865?", "Compare the year numbers.", "Which year is earlier?", "I chose 1865 because it is the larger number.", "The event in 1776 came first because 1776 is earlier than 1865.", "1776 came first"),
        c("industrialization", "Industrialization shifts production toward machines and factories and often increases urban growth as workers move near jobs.", "Why did many cities grow during industrialization?", "Connect factory locations with where workers needed to live.", "Where would workers move to find factory jobs?", "I said cities grew because farms needed more workers.", "Many people moved to cities for factory jobs, increasing urban populations.", "increasing urban populations"),
        c("constitutional rights", "Constitutional rights limit government power and protect individual freedoms, but courts interpret how rights apply in specific conflicts.", "Why are court cases important for constitutional rights?", "Consider who decides how broad language applies to real disputes.", "Which institution interprets constitutional language in specific cases?", "I said rights never need interpretation because the words answer every case.", "Court cases interpret how constitutional protections apply to specific facts and conflicts.", "apply to specific facts"),
    ],
}


LEVELS = ("late_grade_school", "middle_school", "high_school")
IRRELEVANT_PASSAGES = {
    "math": "Plant cells have cell walls outside their cell membranes.",
    "science": "A complete sentence contains a subject and a predicate.",
    "english": "Parallel lines in a plane never intersect.",
    "social_studies": "Condensation changes water vapor into liquid droplets.",
}


def routed_prompt(passage: str, conversation: str, question: str) -> str:
    previous = f"\nPREVIOUS GROUNDED EXCHANGE:\n{conversation}\n" if conversation else ""
    return f"""Decide whether the textbook passage directly supports answering the question.
Your first output must be exactly one route marker:
- [TEXTBOOK] if the passage supports the answer. Answer using the passage.
- [GENERAL] if it does not. Answer from general knowledge.
Never output both markers. Put the answer immediately after the marker.

TEXTBOOK PASSAGE:
{passage}
{previous}
{ANSWER_INSTRUCTION}
Question: {question}"""


def row(concept_id: str, subject: str, level: str, split: str, move: str,
        route: str, prompt: str, response: str, reveal: bool,
        must_include: list[str] | None = None,
        must_not_include: list[str] | None = None) -> dict:
    return {
        "id": f"{concept_id}-{move}",
        "concept_id": concept_id,
        "split": split,
        "subject": subject,
        "level": level,
        "route": route,
        "tutor_move": move,
        "answer_reveal_allowed": reveal,
        "evaluation": {
            "must_include": must_include or [],
            "must_not_include": must_not_include or [],
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def split_for(subject_index: int, concept_index: int) -> str:
    if concept_index < 12:
        return "train"
    if subject_index < 2:
        return "validation" if concept_index < 14 else "test"
    return "validation" if concept_index == 12 else "test"


def make_rows() -> list[dict]:
    rows: list[dict] = []
    for subject_index, (subject, concepts) in enumerate(SUBJECTS.items()):
        for concept_index, concept in enumerate(concepts):
            concept_id = f"{subject[:3]}-{concept_index + 1:02d}"
            level = LEVELS[concept_index % len(LEVELS)]
            split = split_for(subject_index, concept_index)
            textbook_prompt = routed_prompt(concept["passage"], "", concept["exercise"])
            unrelated = IRRELEVANT_PASSAGES[subject]

            rows.append(row(
                concept_id, subject, level, split, "diagnostic", "TEXTBOOK",
                routed_prompt(concept["passage"], "", f"{concept['exercise']} I tried this: {concept['wrong']}"),
                f"[TEXTBOOK]\nLet's inspect your approach before finishing it. {concept['hint']} {concept['check']}",
                False, must_not_include=[concept["answer_phrase"]],
            ))
            rows.append(row(
                concept_id, subject, level, split, "initial_hint", "TEXTBOOK",
                textbook_prompt,
                f"[TEXTBOOK]\n{concept['hint']} {concept['check']}",
                False, must_not_include=[concept["answer_phrase"]],
            ))
            rows.append(row(
                concept_id, subject, level, split, "concept_explanation", "TEXTBOOK",
                routed_prompt(concept["passage"], "", f"Explain {concept['title']} in simple terms."),
                f"[TEXTBOOK]\n{concept['passage']} {concept['check']}",
                True,
            ))
            rows.append(row(
                concept_id, subject, level, split, "corrective_feedback", "GENERAL",
                routed_prompt(unrelated, "", f"{concept['exercise']} My attempt was: {concept['wrong']} Give me feedback, but not the final answer."),
                f"[GENERAL]\nThat approach needs correction. {concept['hint']} {concept['check']}",
                False, must_not_include=[concept["answer_phrase"]],
            ))
            rows.append(row(
                concept_id, subject, level, split, "complete_answer", "GENERAL",
                routed_prompt(
                    unrelated,
                    f"Previous question: {concept['exercise']}\nPrevious answer: {concept['hint']} {concept['check']}",
                    f"I tried this: {concept['wrong']} Please show me the complete answer now.",
                ),
                f"[GENERAL]\n{concept['solution']} Which step in that solution corrected your original approach?",
                True, must_include=[concept["answer_phrase"]],
            ))
    return rows


def main() -> None:
    rows = make_rows()
    for split in ("train", "validation", "test"):
        path = ROOT / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in rows:
                if item["split"] == split:
                    handle.write(json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n")
        print(f"wrote {sum(item['split'] == split for item in rows)} rows to {path}")


if __name__ == "__main__":
    main()

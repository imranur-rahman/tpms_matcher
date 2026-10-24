# TPMS Matcher

Ranks the members of a program committee by how well their published work matches a submitted paper, so that chairs can pick reviewers.

## Language

**Venue edition**:
One year of one conference, for example ICSE 2024. Each venue edition has its own program committee and its own submission deadline.
_Avoid_: conference (when the year matters), event

**PC member**:
A person on the program committee list of one venue edition.
_Avoid_: reviewer (for a person who has no assignment), member, candidate

**Reviewer**:
A PC member whom the venue assigned to review a specific submission.
_Avoid_: PC member (when the assignment matters), referee

**Submission**:
The paper that the PC members are ranked against.
_Avoid_: query paper, input paper

**Author PC member**:
A PC member who is also an author of the submission. The benchmark treats author PC members as the correct answer.
_Avoid_: self-match, ground-truth reviewer

**Identity**:
The link between a PC member and one person record in a publication source, such as a DBLP person or an OpenAlex author.
_Avoid_: author match, profile match

**Profile**:
The set of publications that represents the expertise of one PC member.
_Avoid_: publication list, reviewer profile

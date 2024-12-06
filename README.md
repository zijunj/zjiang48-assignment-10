# Assignment 10: Image Search
--------------------

**Goal:** Implement a simplified version of Google Image Search.

This is a GUI assignment. Please set up a GitHub repo like previous assignments. There is no GUI template, but we encourage you to reuse code from earlier homeworks.

**Deliverables:**
 - A GitHub repo containing all the code necessary to reproduce your results, including a `Makefile`.
 - A 1-2 minute video demonstrating the requirements.

**Requirements:**
 - The user can input a text query. After hitting the search button, the user should see the top 5 most relevant images from the database along with their similarity scores.
 - The user can upload an image query. Again, return the top 5 relevant images along with similarity scores.
 - The user can upload both an image and text query. In this case, the user can enter a value between 0.0 and 1.0 indicating how much to weigh the text query relative to the image query. Again, return the top 5 relevant images along with similarity scores.
 - The user can choose to use embeddings corresponding to the first k principle components instead of CLIP embeddings for image queries. You can reuse work from this week's lab.

[This video](https://youtu.be/U2Ga0ydCfNA) should clarify the requirements. (This video doesn't include the last requirement, but you should include it.)

**Please complete the Python notebook as a first step. You don't need to hand this in, but it walks you through implementation steps.**

**Please download these files:**
- [`coco_images_resized.zip`](https://drive.google.com/file/d/1eNQIUlIKqOg-3e205YIMyUnfTTaOIspP/view?usp=sharing): The image files.
- [`image_embeddings.pickle`](https://drive.google.com/file/d/1M0LodmtqPW-WfEUT50iAx9kAqBUo4CWm/view?usp=sharing): The image embeddings.
- [`house.jpg`](https://drive.google.com/file/d/1uXzWnWgGIqwgEGWbWYY2xUtYXVVRvhBd/view?usp=sharing): An example image.


serve:
	jupyter nbconvert --to=slides --reveal-prefix=reveal.js --post=serve Lecture.ipynb

slides:
	jupyter nbconvert --to=slides --reveal-prefix=reveal.js Lecture.ipynb
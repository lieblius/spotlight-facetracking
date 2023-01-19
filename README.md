# spotlight-facetracking

We aimed to make a face tracker that keeps a spotlight on a user. We accomplished this by obtaining an initial bounding box, applying masks, and then using the mean-shift algorithm to update the bounding box. This [report](442_final_report.pdf) aims to describe our methods in approaching, implementing, and testing this project.

Yes we danced in the presentation, I'm cringing a little but I'll leave the demo below.

<p align="center">
  <img src="https://media.giphy.com/media/CqNRw9ewWK4YsG0iLL/giphy.gif"/>
</p>

Setup 
```
pip install -r requirements.txt
python main.py
```

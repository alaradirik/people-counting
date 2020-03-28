from dlib import correlation_tracker, rectangle


class CorrelationTracker:

    count = 0
    def __init__(self, bbox, img):
        self.tracker = correlation_tracker()
        self.tracker.start_track(img,rectangle(int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])))
        self.confidence = 0. 

        self.time_since_update = 0
        self.id = CorrelationTracker.count
        CorrelationTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self, img):
        self.confidence = self.tracker.update(img)

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state()

    def update(self, bbox, img):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Re-start the tracker with detected positions
        if bbox != []:
            self.tracker.start_track(img, rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))


    def get_state(self):
        pos = self.tracker.get_position()
        return [pos.left(), pos.top(),pos.right(),pos.bottom()]
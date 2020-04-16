import numpy as np


# latMid = 42.52578012378177
# m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * latMid) + 1.175 * np.cos(4 * latMid)
# m_per_deg_lon = 111132.954 * np.cos(latMid)


def m2lat(d):
    m_per_deg_lat = 111679.27
    return d / m_per_deg_lat


def lat2m(d):
    m_per_deg_lat = 111679.27
    return d * m_per_deg_lat


def m2lon(d):
    m_per_deg_lon = 12672.57
    return d / m_per_deg_lon


def lon2m(d):
    m_per_deg_lon = 12672.57
    return d * m_per_deg_lon


def distance_gps(gps1, gps2):
    difference_lat = (lat2m(gps1[0] - gps2[0])) ** 2
    difference_lon = (lon2m(gps1[1] - gps2[1])) ** 2
    return np.sqrt(difference_lat + difference_lon)


def distance_encoding(encoding1, encoding2):
    return np.sqrt(np.sum((encoding1 - np.array(encoding2)) ** 2, axis=1))


class EKF:
    def __init__(self, x, P):
        self.x = x
        self.P = P

    def predict(self, f, F, Q):
        self.x = f(self.x)
        self.P = F.dot(self.P).dot(F.T) + Q

    def update(self, z, H, R):
        y = z - self.x[:2]
        S = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        self.P = (np.eye(self.P.shape[0]) - K.dot(H)).dot(self.P)


class TrackList:
    def __init__(self, encoding=False):
        self.tracks = {}
        self.last_id = 0
        if encoding:
            from EmbeddingNet.embedding_net.model_emb import EmbeddingNet  # noqa
            from EmbeddingNet.embedding_net.utils_emb import parse_net_params  # noqa

            EmbeddingNet_path = "/home/marc/M6/M6_VA/EmbeddingNet/"
            weights_path = (
                EmbeddingNet_path
                + "work_dirs/ai_city_challenge/weights/best_model_resnet50_ai_city.h5"
            )
            encodings_path = (
                EmbeddingNet_path
                + "work_dirs/ai_city_challenge/encodings/encodings_resnet50_ai_city.pkl"
            )
            config = EmbeddingNet_path + "configs/ai_city_challenge_resnext50.yml"
            cfg_params = parse_net_params(config)
            self.model_embedding = EmbeddingNet(cfg_params, training=False)
            self.model_embedding.load_model(weights_path)
            self.model_embedding.load_encodings(encodings_path)

    def add(self, track):
        self.last_id += 1
        track.id = self.last_id
        self.tracks[track.id] = track

    def remove(self, track_id):
        del self.tracks[track_id]

    def predict(self, time):
        for id in self.tracks:
            self.tracks[id].predict(time)

    def get_list(self):
        out = []
        for track_id in self.tracks.keys():
            state = self.tracks[track_id].filter.x
            out.append([[state[0], state[1]], track_id])
        return out

    def process_detections(self, detections, time, image=None):
        # If image is passed we use the embeddings if not, without embeddings
        self.predict(time)
        gps_new_ids = []
        id_map = {}
        for detection in detections:
            measurement = detection[0]
            # id = detection[1]
            if image is not None:  # Using encoding
                bbox = detection[2]  # TODO get bbox of each track
                l, t, w, h = bbox.ltwh
                image_car = image[int(t) : int(t + h), int(l) : int(l + w), :]
                encoding = self.model_embedding.predict_encoding(image_car)
            else:
                encoding = None

            track = self.assign_measurement_to_tracks(
                measurement=measurement, encoding=encoding, threshold_distance=7
            )
            if track is None:
                # print(f"detection {detection[1]} not assigned, creating track...")
                track = self.new_track(
                    x=measurement[0], y=measurement[1], encoding=encoding, time=time
                )
            else:
                # print(f"detection {detection[1]} assigned to track {track.id}")
                track.update(z=measurement, encoding=encoding)
            gps_new_ids.append([measurement, track.id])
            id_map[detection[1]] = track.id
        self.track_remover(time)
        return id_map, gps_new_ids

    def track_remover(self, time):
        max_covariance = 30
        time_to_remove = 2
        keys = list(self.tracks.keys())
        for track_id in keys:
            track = self.tracks[track_id]
            # print(f"Checking to remove {track_id}. time={time},  last_time={track.last_time_associated}")
            if (
                time - track.last_time_associated > time_to_remove
                or np.max(track.filter.P[1, 1]) > lat2m(max_covariance)
                or np.max(track.filter.P[2, 2]) > lon2m(max_covariance)
            ):
                # print(f"Removed Track {track_id}!!")
                self.remove(track_id)

    def new_track(self, x, y, time, P=None, vx=0, vy=0, encoding=None):
        if P is None:
            P = np.eye(4)
            P[0][0] = m2lat(5)
            P[1][1] = m2lon(5)
            P[2][2] = m2lat(100)
            P[3][3] = m2lon(100)
        state = np.array([x, y, vx, vy])
        track = Track(state, P, time, encoding)
        self.add(track)
        return track

    def assign_measurement_to_tracks(
        self, measurement, encoding=None, threshold_distance=2
    ):
        # measurement = (x,y)
        min_distance = 9999999
        for t_id in self.tracks:
            state = self.tracks[t_id].filter.x
            encoding_track = self.tracks[t_id].encoding
            if encoding is None:
                distance = distance_gps(state, measurement)
                # distance = (state[0] - measurement[0]) ** 2 + (state[1] - measurement[1]) ** 2
                # print(f"measurement: {measurement}, state: {state[:2]}")
                # print(f"distance: {distance}, minimum distance: {threshold_distance}")
            else:
                alpha = 0.5
                beta = 1
                distance = alpha * distance_gps(
                    state, measurement
                ) + beta * distance_encoding(encoding_track, encoding)
            if distance < min_distance:
                min_distance = distance
                min_track_id = t_id
        if min_distance < threshold_distance:
            return self.tracks[min_track_id]
        else:
            return None


class Track:
    def __init__(self, x, P, time, encoding):
        self.id = -1
        self.filter = EKF(x=x, P=P)
        self.last_time = time
        self.last_time_associated = time
        self.encoding = encoding

    def predict(self, time):
        delta_t = time - self.last_time
        self.last_time = time

        def f(x):
            out = np.copy(x)
            out[0] += delta_t * out[2]
            out[1] += delta_t * out[3]
            return out

        F = np.eye(4)
        F[0][2] = delta_t
        F[1][3] = delta_t
        Q = np.eye(4)
        Q[0][0] = m2lat(2)
        Q[1][1] = m2lon(2)
        Q[2][2] = m2lat(1)
        Q[3][3] = m2lon(1)
        Q = Q * delta_t
        self.filter.predict(f=f, F=F, Q=Q)

    def update(self, z, encoding=None):
        H = np.eye(2, 4)
        R = np.eye(2)
        R[0][0] = 2
        R[1][1] = 2
        self.filter.update(z=z, H=H, R=R)
        self.last_time_associated = self.last_time
        if encoding is not None:
            gamma = 0.8
            self.encoding = gamma * self.encoding + (1 - gamma) * encoding

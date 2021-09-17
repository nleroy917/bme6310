import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
import random
import string

class Spotify():
    """
    Custom python class that leverages the spotipy python package to interface the spotify API and perform custom actions and data manipulations
    """

    def __init__(self, client_id: str, client_secret: str):
        """
        Create's an instance of the custom Spotify class. For this application, no user data is required, this only the client crednetials flow is necessary.
        :param client_id: a valid Client ID obtained from Spotify's developer dashboard
        :param client_secret: a valid Client Secret obtained from Spotify's developer dashboard
        "
        """
        self._client_id = client_id
        self._client_secret = client_secret
        self._spotify = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=self._client_id,
                client_secret=self._client_secret))
        
    def _grouper(self, list: list, n: int) -> list:
        """Groups a list into a list of subllists of length n"""
        return [list[i:i+n] for i in range(0, len(list), n)]

    def get_random_tracks(self, n: int = 1):
        """
        This function will be used to get a random track from the Spotify API.
        I am folloiwng the generic model laid out in this medium article: https://medium.com/@perryjanssen/getting-random-tracks-using-the-spotify-api-61889b0c0c27
        Steps involved:
        1.) generate a random character
        2.) randomly decide to either a.) place wildcard after character, OR b.) wrap character in wildcards
        3.) genearte random offset page to select from
        :return track: a track object from the spotify API
        """
        char = random.choice(string.ascii_lowercase)
        wild_choice = random.choice([0, 1])
        offset = random.choice(range(500))

        if wild_choice:
            q = char + '%'
        else:
            q = '%' + char + '%'

        return self._spotify.search(q=q, offset=offset, type='track', limit=n)['tracks']['items']

    def artists_from_tracks(self, tracks: list[dict]):
        """
        Since genre info is not track specific, rather, artist specific,
        we need a function that returns a list of genres for each track given
        a list of track objects.
        
        For each track, extract out the artist id's... and then get artist
        object from the API, extract out the genres and append then return
        """
        artist_ids = [
            # extract out first artist only for simplicity
            t['artists'][0]['id'] for t in tracks
        ]
        
        all_artists = []
        if len(artist_ids) > 50:
            grouped_ids = self._grouper(artist_ids, 50)
            for group in grouped_ids:
                all_artists += self._spotify.artists(group)['artists']
        else:
            all_artists += self._spotify.artists(artist_ids)['artists']
        
        return all_artists
    
    def audio_features(self, tracks: list[dict]):
        """
        Get audio features for several tracks
        :param ids: a list of track ids
        """
        track_ids = [
            t['id'] for t in tracks
        ]
        all_features = []
        if len(track_ids) > 100:
            grouped_ids = self._grouper(track_ids, 100)
            for group in grouped_ids:
                all_features += self._spotify.audio_features(group)
        else:
            all_features += self._spotify.audio_features(track_ids)
        
        return all_features

if __name__ == '__main__':
    import pandas as pd
    from progress.bar import ChargingBar
    from dotenv import load_dotenv
    import os
    load_dotenv()

    sp = Spotify(
        client_id=os.getenv('CLIENT_ID'),
        client_secret=os.getenv('CLIENT_SECRET')
    )
    
    # randomly sample many tracks from 
    # the spotify servers
    N = 5000
    all_tracks = []
    all_ids = []
    
    # progress bar
    b = ChargingBar('Collecting tracks', max=N, suffix=f"%(index)d/{N}")
    print('Starting collection.')
    
    # gather in groups of 10
    while len(all_tracks) < N:
        try:
            tracks = sp.get_random_tracks(n=50)
        except SpotifyException as se:
            print(f"Error reached {se}. skipping...")
        except KeyboardInterrupt as ki:
            print("\nStopping...")
            break
            
        for t in tracks:
            if t['id'] not in all_ids:
                all_tracks.append(t)
                all_ids.append(t['id'])
                b.next()
    # end progress
    b.finish()
    
    print("Fetching artist data...")
    # link artists to tracks
    all_artists = sp.artists_from_tracks(all_tracks)
    for i in range(len(all_tracks)):
        all_tracks[i]['artist'] = all_artists[i]
    
    print("Fetching analysis data...")
    # gather audio feature data
    # and link to tracks
    all_features = sp.audio_features(all_tracks)
    for i in range(len(all_tracks)):
        all_tracks[i]['analysis'] = all_features[i]
    
    print("Linking data...")
    # we should now have all of our data and can
    # link it to a dataframe and dump to csv
    data = []
    for track in all_tracks:
        if track['analysis'] is not None:
            data.append({
                'name': track['name'],
                'id': track['id'],
                'artist': track['artist']['name'],
                'genre': track['artist']['genres'][0] if len(track['artist']['genres']) > 0 else None, # take the first...
                "acousticness": track['analysis']['acousticness'],
                "danceability": track['analysis']['danceability'],
                "energy": track['analysis']['energy'],
                "instrumentalness": track['analysis']['instrumentalness'],
                "liveness": track['analysis']['liveness'],
                "loudness": track['analysis']['loudness'],
                "speechiness": track['analysis']['speechiness'],
                "tempo": track['analysis']['tempo'],
                "valence": track['analysis']['valence'],
            })
           
    print("Exporting to csv...")
    df = pd.DataFrame(data)
    df.to_csv('data.csv')
    print("Done.")
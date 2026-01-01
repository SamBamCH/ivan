from ivan.sports.nfl.refactor.game import generate_correlated_samples
from ivan.sports.nfl.refactor.lineup import generate_lineups_subprocess
from multiprocessing import Process, freeze_support, Queue


def _detect_slate_type(players):
    roster_slot_ids = []
    for p in players:
        if p.roster_slot_id not in roster_slot_ids:
            roster_slot_ids.append(p.roster_slot_id)
    if any(x in roster_slot_ids for x in (511, 512)):
        return 'showdown'
    else:
        return 'classic'



class Slate:
    def __init__(self, players, session, n_cores):
        self.slate_type = _detect_slate_type(players)
        self.players = players
        self.session = session
        self.n_cores = n_cores
        

    def generate_lineups_for_slate(self, num_lineups):
        q = Queue()
        #generate samples for all players
        samples = generate_correlated_samples(self.session, self.players, n_samples=1000, randomness=1)
        print(f'generated {len(samples)} samples, moving on to PuLP....')
        #prepare parallel jobs
        base = num_lineups // self.n_cores
        remainder = num_linueps % self.n_cores
        quotas = [base + (1 if i < remainder else 0) for i in range(self.n_cores)]
        #generate_lineup_subprocess
        procs = []
        for proc_no, qnty in enumerate(quotas):
            p = Process(
                target=generate_lineups_subprocess,
                args=(self.session, self.players, samples, qnty)
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        q.close(); q.join_thread()
        #deduplicate
        #visualize lineups
        #set limit/randomness
        #regenerate
        #deduplicate
        #visualize
        #loop as much as we want until we pass (like NBA).
    

    #def simulate_contests_for_slate(self):



    #def choose_contests_for_slate(self):


    

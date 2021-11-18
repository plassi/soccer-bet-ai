import numpy as np
import pandas as pd

class Simulation:

    def run(self, dictionary, predictions):

        start_cash = 100
        value_limit = 1.2
        odds_limit = 3.3
        bet_amount = 1

        cash = pd.Series([start_cash])
        all_bets = []
        all_bets_outcomes = pd.Series()
        for i in range(len(dictionary)):
            bet_values = []

            fixture = dictionary[i]
            game_result = fixture['result']
            winning_odds = [float(fixture['winner_odds']['home']),
                            float(fixture['winner_odds']['draw']),
                            float(fixture['winner_odds']['away'])]
            predicted_odds = [1 / predictions[i][0][0],
                              1 / predictions[i][0][1],
                              1 / predictions[i][0][2]]
            bet_values = [winning_odds[0] / predicted_odds[0],
                          winning_odds[1] / predicted_odds[1],
                          winning_odds[2] / predicted_odds[2]]
            bet = None
            bets = []

            # Select bet
            for i, value in enumerate(bet_values):
                if value > value_limit and float(predicted_odds[i]) < odds_limit:
                    bets.append(i)
            if len(bets) > 1:
                # select the bet with lower predicted odds
                selected_odds = [predicted_odds[i].item() for i in bets]
                bet = bets[np.argmin(selected_odds)]
                all_bets.append(bet)
            elif len(bets) == 1:
                bet = bets[0]
                all_bets.append(bet)

            # print(fixture)
            # print(winning_odds)
            # print(predicted_odds)
            # print(bet_values)
            # print(bet)

            # Make bet
            if bet is not None:
                if game_result == bet:
                    cash_result = bet_amount * winning_odds[bet] - bet_amount
                    all_bets_outcomes = all_bets_outcomes.append(pd.Series([cash_result]))
                    # print(cash.values[-1])
                    cash = cash.append(pd.Series([cash.values[-1] + cash_result]))
                else:
                    cash = cash.append(pd.Series([cash.values[-1] - bet_amount]))
                    all_bets_outcomes = all_bets_outcomes.append(pd.Series([-bet_amount]))

        # print(cash)

        previous_peaks = pd.DataFrame(cash).cummax()

        # print(previous_peaks)


        drawdown = (cash - previous_peaks) / previous_peaks

        # return cash, max_drawdown
        return cash, drawdown, all_bets, all_bets_outcomes

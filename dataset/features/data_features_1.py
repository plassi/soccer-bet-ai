class DataFeatures1:

	# For loader, load only where fixture_status_short == 'FT'

	all_features = [

		'fixture_referee', # 'D.Schlager' : one-hot-encode
		'fixture_timestamp', # 1618745400 : normalize
		'fixture_venue_name', # 'St. Paul\'s Stadium' : one-hot-encode
		'fixture_status_short', # 'FT' : skip
		'league_id', # 79 : one-hot-encode
		'league_country', # 'Germany' : one-hot-encode
		'league_season', # '2020' : one-hot-encode
		'league_round', # 'Regular Season - 29' : one-hot-encode
		'teams_home_id', # '18' : one-hot-encode
		'teams_home_winner', # True, False, null : skip, to create targets
		'teams_away_id', # '18' : one-hot-encode

		'predictions_0_predictions_winner_id', # '18' : one-hot-encode
		'predictions_0_predictions_win_or_draw', # True, False : one-hot-encode
		'predictions_0_predictions_under_over', # -3.5, 1.5, null : normalize
		'predictions_0_predictions_goals_home', # -3.5, -1.5, null : normalize
		'predictions_0_predictions_goals_away', # -3.5, -1.5, null : normalize

		'predictions_0_predictions_percent_home', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_predictions_percent_draw', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_predictions_percent_away', # '45%', '10%': change to 0.45, 0.10, normalize

		'predictions_0_teams_home_last_5_form', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_last_5_att', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_last_5_def', # '45%', '10%': change to 0.45, 0.10, normalize

		'predictions_0_teams_home_last_5_goals_for_total', # 5, 7, 14 : normalize
		'predictions_0_teams_home_last_5_goals_for_average', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_home_last_5_goals_against_total', # 5, 7, 14 : normalize
		'predictions_0_teams_home_last_5_goals_against_average', # 1.4, 1, 0.6 : normalize

		'predictions_0_teams_home_league_form', # 'DWDLWWLWWLWLWWWWDWLWWLWWLWWL' : reverse, find longest string in column. Separate letters to own columns. One-hot-encode.

		'predictions_0_teams_home_league_fixtures_played_home', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_played_away', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_played_total', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_wins_home', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_wins_away', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_wins_total', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_draws_home', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_draws_away', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_draws_total', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_loses_home', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_loses_away', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_loses_total', # 10, 22, : normalize
		'predictions_0_teams_home_league_goals_for_total_home', # 10, 22, : normalize
		'predictions_0_teams_home_league_goals_for_total_away', # 10, 22, : normalize
		'predictions_0_teams_home_league_goals_for_total_total', # 10, 22, : normalize

		'predictions_0_teams_home_league_goals_for_average_home', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_home_league_goals_for_average_away', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_home_league_goals_for_average_total', # 1.4, 1, 0.6 : normalize

		'predictions_0_teams_home_league_goals_for_minute_0-15_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_for_minute_0-15_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_home_league_goals_for_minute_16-30_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_for_minute_16-30_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_home_league_goals_for_minute_31-45_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_for_minute_31-45_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_home_league_goals_for_minute_46-60_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_for_minute_46-60_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_home_league_goals_for_minute_61-75_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_for_minute_61-75_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_home_league_goals_for_minute_76-90_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_for_minute_76-90_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_home_league_goals_for_minute_91-105_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_for_minute_91-105_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		
		'predictions_0_teams_home_league_goals_against_total_home', # 12, 23, 33, null : normalize
		'predictions_0_teams_home_league_goals_against_total_away', # 13, 23, 33, null : normalize
		'predictions_0_teams_home_league_goals_against_total_total', # 13, 23, 33, null : normalize

		'predictions_0_teams_home_league_goals_against_average_home', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_home_league_goals_against_average_away', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_home_league_goals_against_average_total', # 1.4, 1, 0.6 : normalize

		'predictions_0_teams_home_league_goals_against_minute_0-15_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_0-15_percentage', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_16-30_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_16-30_percentage', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_31-45_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_31-45_percentage', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_46-60_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_46-60_percentage', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_61-75_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_61-75_percentage', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_76-90_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_76-90_percentage', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_91-105_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_91-105_percentage', # 1, 2, 3, null : normalize

		'predictions_0_teams_home_league_biggest_streak_wins', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_biggest_streak_draws', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_biggest_streak_loses', # 1, 2, 3, null : normalize

		'predictions_0_teams_home_league_biggest_wins_home', # '3-0', separate to 3, 0, normalize
		'predictions_0_teams_home_league_biggest_wins_away', # '0-3', separate to 3, 0, normalize
		'predictions_0_teams_home_league_biggest_loses_home', # '0-3', separate to 3, 0, normalize
		'predictions_0_teams_home_league_biggest_loses_away', # '3-0', separate to 3, 0, normalize

		'predictions_0_teams_home_league_biggest_goals_for_home', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_biggest_goals_for_away', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_biggest_goals_against_home', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_biggest_goals_against_away', # 1, 2, 3, null : normalize

		'predictions_0_teams_home_league_clean_sheet_home', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_clean_sheet_away', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_clean_sheet_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_failed_to_score_home', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_failed_to_score_away', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_failed_to_score_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_penalty_scored_total', # 1, 2, 3, null : normalize

		'predictions_0_teams_home_league_penalty_scored_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_penalty_missed_total', # 0, 1, 2, null : normalize
		'predictions_0_teams_home_league_penalty_missed_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_penalty_total', # 0, 5, 7, null : normalize

		'predictions_0_teams_home_league_cards_yellow_0-15_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_yellow_0-15_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_yellow_16-30_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_yellow_16-30_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_yellow_31-45_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_yellow_31-45_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_yellow_46-60_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_yellow_46-60_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_yellow_61-75_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_yellow_61-75_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_yellow_76-90_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_yellow_76-90_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_yellow_91-105_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_yellow_91-105_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_red_0-15_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_red_0-15_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_red_16-30_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_red_16-30_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_red_31-45_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_red_31-45_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_red_46-60_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_red_46-60_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_red_61-75_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_red_61-75_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_red_76-90_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_red_76-90_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_red_91-105_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_red_91-105_percentage', # '45%', '10%': change to 0.45, 0.10, normalize

		'predictions_0_teams_away_last_5_form',
		'predictions_0_teams_away_last_5_att',
		'predictions_0_teams_away_last_5_def',
		'predictions_0_teams_away_last_5_goals_for_total',
		'predictions_0_teams_away_last_5_goals_for_average',
		'predictions_0_teams_away_last_5_goals_against_total',
		'predictions_0_teams_away_last_5_goals_against_average',
		'predictions_0_teams_away_league_form',
		'predictions_0_teams_away_league_fixtures_played_home',
		'predictions_0_teams_away_league_fixtures_played_away',
		'predictions_0_teams_away_league_fixtures_played_total',
		'predictions_0_teams_away_league_fixtures_wins_home',
		'predictions_0_teams_away_league_fixtures_wins_away',
		'predictions_0_teams_away_league_fixtures_wins_total',
		'predictions_0_teams_away_league_fixtures_draws_home',
		'predictions_0_teams_away_league_fixtures_draws_away',
		'predictions_0_teams_away_league_fixtures_draws_total',
		'predictions_0_teams_away_league_fixtures_loses_home',
		'predictions_0_teams_away_league_fixtures_loses_away',
		'predictions_0_teams_away_league_fixtures_loses_total',
		'predictions_0_teams_away_league_goals_for_total_home',
		'predictions_0_teams_away_league_goals_for_total_away',
		'predictions_0_teams_away_league_goals_for_total_total',
		'predictions_0_teams_away_league_goals_for_average_home',
		'predictions_0_teams_away_league_goals_for_average_away',
		'predictions_0_teams_away_league_goals_for_average_total',
		'predictions_0_teams_away_league_goals_for_minute_0-15_total',
		'predictions_0_teams_away_league_goals_for_minute_0-15_percentage',
		'predictions_0_teams_away_league_goals_for_minute_16-30_total',
		'predictions_0_teams_away_league_goals_for_minute_16-30_percentage',
		'predictions_0_teams_away_league_goals_for_minute_31-45_total',
		'predictions_0_teams_away_league_goals_for_minute_31-45_percentage',
		'predictions_0_teams_away_league_goals_for_minute_46-60_total',
		'predictions_0_teams_away_league_goals_for_minute_46-60_percentage',
		'predictions_0_teams_away_league_goals_for_minute_61-75_total',
		'predictions_0_teams_away_league_goals_for_minute_61-75_percentage',
		'predictions_0_teams_away_league_goals_for_minute_76-90_total',
		'predictions_0_teams_away_league_goals_for_minute_76-90_percentage',
		'predictions_0_teams_away_league_goals_for_minute_91-105_total',
		'predictions_0_teams_away_league_goals_for_minute_91-105_percentage',
		'predictions_0_teams_away_league_goals_against_total_home',
		'predictions_0_teams_away_league_goals_against_total_away',
		'predictions_0_teams_away_league_goals_against_total_total',
		'predictions_0_teams_away_league_goals_against_average_home',
		'predictions_0_teams_away_league_goals_against_average_away',
		'predictions_0_teams_away_league_goals_against_average_total',
		'predictions_0_teams_away_league_goals_against_minute_0-15_total',
		'predictions_0_teams_away_league_goals_against_minute_0-15_percentage',
		'predictions_0_teams_away_league_goals_against_minute_16-30_total',
		'predictions_0_teams_away_league_goals_against_minute_16-30_percentage',
		'predictions_0_teams_away_league_goals_against_minute_31-45_total',
		'predictions_0_teams_away_league_goals_against_minute_31-45_percentage',
		'predictions_0_teams_away_league_goals_against_minute_46-60_total',
		'predictions_0_teams_away_league_goals_against_minute_46-60_percentage',
		'predictions_0_teams_away_league_goals_against_minute_61-75_total',
		'predictions_0_teams_away_league_goals_against_minute_61-75_percentage',
		'predictions_0_teams_away_league_goals_against_minute_76-90_total',
		'predictions_0_teams_away_league_goals_against_minute_76-90_percentage',
		'predictions_0_teams_away_league_goals_against_minute_91-105_total',
		'predictions_0_teams_away_league_goals_against_minute_91-105_percentage',
		'predictions_0_teams_away_league_biggest_streak_wins',
		'predictions_0_teams_away_league_biggest_streak_draws',
		'predictions_0_teams_away_league_biggest_streak_loses',
		'predictions_0_teams_away_league_biggest_wins_home',
		'predictions_0_teams_away_league_biggest_wins_away',
		'predictions_0_teams_away_league_biggest_loses_home',
		'predictions_0_teams_away_league_biggest_loses_away',
		'predictions_0_teams_away_league_biggest_goals_for_home',
		'predictions_0_teams_away_league_biggest_goals_for_away',
		'predictions_0_teams_away_league_biggest_goals_against_home',
		'predictions_0_teams_away_league_biggest_goals_against_away',
		'predictions_0_teams_away_league_clean_sheet_home',
		'predictions_0_teams_away_league_clean_sheet_away',
		'predictions_0_teams_away_league_clean_sheet_total',
		'predictions_0_teams_away_league_failed_to_score_home',
		'predictions_0_teams_away_league_failed_to_score_away',
		'predictions_0_teams_away_league_failed_to_score_total',
		'predictions_0_teams_away_league_penalty_scored_total',
		'predictions_0_teams_away_league_penalty_scored_percentage',
		'predictions_0_teams_away_league_penalty_missed_total',
		'predictions_0_teams_away_league_penalty_missed_percentage',
		'predictions_0_teams_away_league_penalty_total',
		'predictions_0_teams_away_league_cards_yellow_0-15_total',
		'predictions_0_teams_away_league_cards_yellow_0-15_percentage',
		'predictions_0_teams_away_league_cards_yellow_16-30_total',
		'predictions_0_teams_away_league_cards_yellow_16-30_percentage',
		'predictions_0_teams_away_league_cards_yellow_31-45_total',
		'predictions_0_teams_away_league_cards_yellow_31-45_percentage',
		'predictions_0_teams_away_league_cards_yellow_46-60_total',
		'predictions_0_teams_away_league_cards_yellow_46-60_percentage',
		'predictions_0_teams_away_league_cards_yellow_61-75_total',
		'predictions_0_teams_away_league_cards_yellow_61-75_percentage',
		'predictions_0_teams_away_league_cards_yellow_76-90_total',
		'predictions_0_teams_away_league_cards_yellow_76-90_percentage',
		'predictions_0_teams_away_league_cards_yellow_91-105_total',
		'predictions_0_teams_away_league_cards_yellow_91-105_percentage',
		'predictions_0_teams_away_league_cards_red_0-15_total',
		'predictions_0_teams_away_league_cards_red_0-15_percentage',
		'predictions_0_teams_away_league_cards_red_16-30_total',
		'predictions_0_teams_away_league_cards_red_16-30_percentage',
		'predictions_0_teams_away_league_cards_red_31-45_total',
		'predictions_0_teams_away_league_cards_red_31-45_percentage',
		'predictions_0_teams_away_league_cards_red_46-60_total',
		'predictions_0_teams_away_league_cards_red_46-60_percentage',
		'predictions_0_teams_away_league_cards_red_61-75_total',
		'predictions_0_teams_away_league_cards_red_61-75_percentage',
		'predictions_0_teams_away_league_cards_red_76-90_total',
		'predictions_0_teams_away_league_cards_red_76-90_percentage',
		'predictions_0_teams_away_league_cards_red_91-105_total',
		'predictions_0_teams_away_league_cards_red_91-105_percentage',

		'predictions_0_comparison_form_home', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_comparison_form_away', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_comparison_att_home', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_comparison_att_away', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_comparison_def_home', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_comparison_def_away', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_comparison_poisson_distribution_home', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_comparison_poisson_distribution_away', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_comparison_h2h_home', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_comparison_h2h_away', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_comparison_goals_home', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_comparison_goals_away', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_comparison_total_home', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_comparison_total_away', # '45%', '10%': change to 0.45, 0.10, normalize

		'predictions_0_h2h_0_fixture_status_short', # 'FT' one-hot-encode 
		'predictions_0_h2h_0_teams_home_id', # '79' one-hot-encode
		'predictions_0_h2h_0_teams_away_id', # '79' one-hot-encode
		'predictions_0_h2h_0_goals_home', # '1' normalize
		'predictions_0_h2h_0_goals_away', # '0' normalize

		'predictions_0_h2h_1_fixture_status_short', # 'FT' one-hot-encode 
		'predictions_0_h2h_1_teams_home_id', # '79' one-hot-encode
		'predictions_0_h2h_1_teams_away_id', # '79' one-hot-encode
		'predictions_0_h2h_1_goals_home', # '1' normalize
		'predictions_0_h2h_1_goals_away', # '0' normalize

		'predictions_0_h2h_2_fixture_status_short', # 'FT' one-hot-encode 
		'predictions_0_h2h_2_teams_home_id', # '79' one-hot-encode
		'predictions_0_h2h_2_teams_away_id', # '79' one-hot-encode
		'predictions_0_h2h_2_goals_home', # '1' normalize
		'predictions_0_h2h_2_goals_away', # '0' normalize

		'predictions_0_h2h_3_fixture_status_short', # 'FT' one-hot-encode 
		'predictions_0_h2h_3_teams_home_id', # '79' one-hot-encode
		'predictions_0_h2h_3_teams_away_id', # '79' one-hot-encode
		'predictions_0_h2h_3_goals_home', # '1' normalize
		'predictions_0_h2h_3_goals_away', # '0' normalize

		'predictions_0_h2h_4_fixture_status_short', # 'FT' one-hot-encode 
		'predictions_0_h2h_4_teams_home_id', # '79' one-hot-encode
		'predictions_0_h2h_4_teams_away_id', # '79' one-hot-encode
		'predictions_0_h2h_4_goals_home', # '1' normalize
		'predictions_0_h2h_4_goals_away', # '0' normalize
	]

	scores_features = [
		'predictions_0_teams_home_league_biggest_wins_home', # '3-0', separate to 3, 0, normalize
		'predictions_0_teams_home_league_biggest_wins_away', # '0-3', separate to 3, 0, normalize
		'predictions_0_teams_home_league_biggest_loses_home', # '0-3', separate to 3, 0, normalize
		'predictions_0_teams_home_league_biggest_loses_away', # '3-0', separate to 3, 0, normalize
		'predictions_0_teams_away_league_biggest_wins_home', # '3-0', separate to 3, 0, normalize
		'predictions_0_teams_away_league_biggest_wins_away', # '0-3', separate to 3, 0, normalize
		'predictions_0_teams_away_league_biggest_loses_home', # '0-3', separate to 3, 0, normalize
		'predictions_0_teams_away_league_biggest_loses_away', # '3-0', separate to 3, 0, normalize
	]

	form_features = [
		'predictions_0_teams_home_league_form', # 'DWDLWWLWWLWLWWWWDWLWWLWWLWWL' : reverse, find longest string in column. Separate letters to own columns. One-hot-encode.
		'predictions_0_teams_away_league_form', # 'DWDLWWLWWLWLWWWWDWLWWLWWLWWL' : reverse, find longest string in column. Separate letters to own columns. One-hot-encode.
	]

	normalize_features = [
		'predictions_0_predictions_under_over', # -3.5, 1.5, null : normalize
		'predictions_0_predictions_goals_home', # -3.5, -1.5, null : normalize
		'predictions_0_predictions_goals_away', # -3.5, -1.5, null : normalize

		'predictions_0_teams_home_last_5_goals_for_total', # 5, 7, 14 : normalize
		'predictions_0_teams_home_last_5_goals_for_average', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_home_last_5_goals_against_total', # 5, 7, 14 : normalize
		'predictions_0_teams_home_last_5_goals_against_average', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_home_league_fixtures_played_home', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_played_away', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_played_total', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_wins_home', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_wins_away', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_wins_total', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_draws_home', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_draws_away', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_draws_total', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_loses_home', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_loses_away', # 10, 22, : normalize
		'predictions_0_teams_home_league_fixtures_loses_total', # 10, 22, : normalize
		'predictions_0_teams_home_league_goals_for_total_home', # 10, 22, : normalize
		'predictions_0_teams_home_league_goals_for_total_away', # 10, 22, : normalize
		'predictions_0_teams_home_league_goals_for_total_total', # 10, 22, : normalize
		'predictions_0_teams_home_league_goals_for_average_home', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_home_league_goals_for_average_away', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_home_league_goals_for_average_total', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_home_league_goals_for_minute_0-15_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_for_minute_16-30_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_for_minute_31-45_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_for_minute_46-60_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_for_minute_61-75_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_for_minute_76-90_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_for_minute_91-105_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_total_home', # 12, 23, 33, null : normalize
		'predictions_0_teams_home_league_goals_against_total_away', # 13, 23, 33, null : normalize
		'predictions_0_teams_home_league_goals_against_total_total', # 13, 23, 33, null : normalize
		'predictions_0_teams_home_league_goals_against_average_home', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_home_league_goals_against_average_away', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_home_league_goals_against_average_total', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_home_league_goals_against_minute_0-15_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_16-30_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_31-45_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_46-60_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_61-75_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_76-90_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_goals_against_minute_91-105_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_biggest_streak_wins', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_biggest_streak_draws', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_biggest_streak_loses', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_biggest_goals_for_home', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_biggest_goals_for_away', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_biggest_goals_against_home', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_biggest_goals_against_away', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_clean_sheet_home', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_clean_sheet_away', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_clean_sheet_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_failed_to_score_home', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_failed_to_score_away', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_failed_to_score_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_penalty_scored_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_penalty_missed_total', # 0, 1, 2, null : normalize
		'predictions_0_teams_home_league_penalty_total', # 0, 5, 7, null : normalize
		'predictions_0_teams_home_league_cards_yellow_0-15_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_yellow_16-30_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_yellow_31-45_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_yellow_46-60_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_yellow_61-75_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_yellow_76-90_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_yellow_91-105_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_red_0-15_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_red_16-30_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_red_31-45_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_red_46-60_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_red_61-75_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_red_76-90_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_home_league_cards_red_91-105_total', # 1, 2, 3, null : normalize

		'predictions_0_teams_away_last_5_goals_for_total', # 5, 7, 14 : normalize
		'predictions_0_teams_away_last_5_goals_for_average', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_away_last_5_goals_against_total', # 5, 7, 14 : normalize
		'predictions_0_teams_away_last_5_goals_against_average', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_away_league_fixtures_played_home', # 10, 22, : normalize
		'predictions_0_teams_away_league_fixtures_played_away', # 10, 22, : normalize
		'predictions_0_teams_away_league_fixtures_played_total', # 10, 22, : normalize
		'predictions_0_teams_away_league_fixtures_wins_home', # 10, 22, : normalize
		'predictions_0_teams_away_league_fixtures_wins_away', # 10, 22, : normalize
		'predictions_0_teams_away_league_fixtures_wins_total', # 10, 22, : normalize
		'predictions_0_teams_away_league_fixtures_draws_home', # 10, 22, : normalize
		'predictions_0_teams_away_league_fixtures_draws_away', # 10, 22, : normalize
		'predictions_0_teams_away_league_fixtures_draws_total', # 10, 22, : normalize
		'predictions_0_teams_away_league_fixtures_loses_home', # 10, 22, : normalize
		'predictions_0_teams_away_league_fixtures_loses_away', # 10, 22, : normalize
		'predictions_0_teams_away_league_fixtures_loses_total', # 10, 22, : normalize
		'predictions_0_teams_away_league_goals_for_total_home', # 10, 22, : normalize
		'predictions_0_teams_away_league_goals_for_total_away', # 10, 22, : normalize
		'predictions_0_teams_away_league_goals_for_total_total', # 10, 22, : normalize
		'predictions_0_teams_away_league_goals_for_average_home', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_away_league_goals_for_average_away', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_away_league_goals_for_average_total', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_away_league_goals_for_minute_0-15_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_goals_for_minute_16-30_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_goals_for_minute_31-45_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_goals_for_minute_46-60_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_goals_for_minute_61-75_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_goals_for_minute_76-90_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_goals_for_minute_91-105_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_goals_against_total_home', # 12, 23, 33, null : normalize
		'predictions_0_teams_away_league_goals_against_total_away', # 13, 23, 33, null : normalize
		'predictions_0_teams_away_league_goals_against_total_total', # 13, 23, 33, null : normalize
		'predictions_0_teams_away_league_goals_against_average_home', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_away_league_goals_against_average_away', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_away_league_goals_against_average_total', # 1.4, 1, 0.6 : normalize
		'predictions_0_teams_away_league_goals_against_minute_0-15_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_goals_against_minute_16-30_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_goals_against_minute_31-45_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_goals_against_minute_46-60_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_goals_against_minute_61-75_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_goals_against_minute_76-90_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_goals_against_minute_91-105_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_biggest_streak_wins', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_biggest_streak_draws', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_biggest_streak_loses', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_biggest_goals_for_home', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_biggest_goals_for_away', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_biggest_goals_against_home', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_biggest_goals_against_away', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_clean_sheet_home', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_clean_sheet_away', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_clean_sheet_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_failed_to_score_home', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_failed_to_score_away', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_failed_to_score_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_penalty_scored_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_penalty_missed_total', # 0, 1, 2, null : normalize
		'predictions_0_teams_away_league_penalty_total', # 0, 5, 7, null : normalize
		'predictions_0_teams_away_league_cards_yellow_0-15_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_cards_yellow_16-30_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_cards_yellow_31-45_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_cards_yellow_46-60_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_cards_yellow_61-75_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_cards_yellow_76-90_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_cards_yellow_91-105_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_cards_red_0-15_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_cards_red_16-30_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_cards_red_31-45_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_cards_red_46-60_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_cards_red_61-75_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_cards_red_76-90_total', # 1, 2, 3, null : normalize
		'predictions_0_teams_away_league_cards_red_91-105_total', # 1, 2, 3, null : normalize
	]

	percentage_features = [
		'predictions_0_predictions_percent_home', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_predictions_percent_draw', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_predictions_percent_away', # '45%', '10%': change to 0.45, 0.10, normalize

		'predictions_0_teams_home_last_5_form', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_last_5_att', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_last_5_def', # '45%', '10%': change to 0.45, 0.10, normalize

		'predictions_0_teams_home_league_goals_for_minute_0-15_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_home_league_goals_for_minute_16-30_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_home_league_goals_for_minute_31-45_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_home_league_goals_for_minute_46-60_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_home_league_goals_for_minute_61-75_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_home_league_goals_for_minute_76-90_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_home_league_goals_for_minute_91-105_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_home_league_penalty_scored_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_penalty_missed_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_yellow_0-15_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_yellow_16-30_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_yellow_31-45_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_yellow_46-60_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_yellow_61-75_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_yellow_76-90_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_yellow_91-105_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_red_0-15_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_red_16-30_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_red_31-45_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_red_46-60_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_red_61-75_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_red_76-90_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_home_league_cards_red_91-105_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
	
		'predictions_0_teams_away_last_5_form', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_last_5_att', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_last_5_def', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_goals_for_minute_0-15_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_away_league_goals_for_minute_16-30_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_away_league_goals_for_minute_31-45_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_away_league_goals_for_minute_46-60_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_away_league_goals_for_minute_61-75_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_away_league_goals_for_minute_76-90_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_away_league_goals_for_minute_91-105_percentage', # '4.52%', '1.04%': change to 0.0452, 0.10, normalize
		'predictions_0_teams_away_league_penalty_scored_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_penalty_missed_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_cards_yellow_0-15_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_cards_yellow_16-30_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_cards_yellow_31-45_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_cards_yellow_46-60_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_cards_yellow_61-75_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_cards_yellow_76-90_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_cards_yellow_91-105_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_cards_red_0-15_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_cards_red_16-30_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_cards_red_31-45_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_cards_red_46-60_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_cards_red_61-75_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_cards_red_76-90_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
		'predictions_0_teams_away_league_cards_red_91-105_percentage', # '45%', '10%': change to 0.45, 0.10, normalize
	
	]

	one_hot_encode_features = [
		'fixture_referee', # 'D.Schlager' : one-hot-encode
		# 'fixture_venue_name', # 'St. Paul\'s Stadium' : one-hot-encode
		# 'league_id', # 79 : one-hot-encode
		# 'league_country', # 'Germany' : one-hot-encode
		'league_season', # '2020' : one-hot-encode
		'league_round', # 'Regular Season - 29' : one-hot-encode
		'teams_home_id', # '18' : one-hot-encode
		'teams_away_id', # '18' : one-hot-encode

		'predictions_0_predictions_winner_id', # '18' : one-hot-encode
		'predictions_0_predictions_win_or_draw', # True, False : one-hot-encode
		'predictions_0_h2h_0_fixture_status_short', # 'FT' one-hot-encode 
		'predictions_0_h2h_0_teams_home_id', # '79' one-hot-encode
		'predictions_0_h2h_0_teams_away_id', # '79' one-hot-encode
		'predictions_0_h2h_1_fixture_status_short', # 'FT' one-hot-encode 
		'predictions_0_h2h_1_teams_home_id', # '79' one-hot-encode
		'predictions_0_h2h_1_teams_away_id', # '79' one-hot-encode
		'predictions_0_h2h_2_fixture_status_short', # 'FT' one-hot-encode 
		'predictions_0_h2h_2_teams_home_id', # '79' one-hot-encode
		'predictions_0_h2h_2_teams_away_id', # '79' one-hot-encode
		'predictions_0_h2h_3_fixture_status_short', # 'FT' one-hot-encode 
		'predictions_0_h2h_3_teams_home_id', # '79' one-hot-encode
		'predictions_0_h2h_3_teams_away_id', # '79' one-hot-encode
		'predictions_0_h2h_4_fixture_status_short', # 'FT' one-hot-encode 
		'predictions_0_h2h_4_teams_home_id', # '79' one-hot-encode
		'predictions_0_h2h_4_teams_away_id', # '79' one-hot-encode
	]


import copy
import unittest

import numpy as np
import pandas as pd

from lab1_rich import causal_rich, post_values


class RichHistoryTests(unittest.TestCase):
    def posts(self):
        return [
            {
                "urn": f"POST_{i}",
                "author": {"code": "AUTHOR_1"},
                "postedDateTimestamp": i * 86400000,
                "totalReactionCount": 5 * i,
                "commentsCount": i,
                "repostsCount": 2 * i,
                "contentType": "image",
                "image": [{"width": 800, "height": 600}],
                "images": [[{"width": 800, "height": 600}]],
                "companyMentions": [{"code": "ORG_1"}],
            }
            for i in range(1, 9)
        ]

    def test_dimensions_read_from_release_field(self):
        values, _, _ = post_values(self.posts()[0])
        self.assertEqual(values["actual_image_count"], 1)
        self.assertAlmostEqual(values["actual_image_aspect"], 4 / 3)
        self.assertAlmostEqual(values["actual_image_log_area"], np.log1p(480000))

    def test_current_and_future_counters_do_not_change_features(self):
        original = self.posts()
        modified = copy.deepcopy(original)
        for post in modified[5:]:
            post["totalReactionCount"] = 1000000
            post["commentsCount"] = 1000000
            post["repostsCount"] = 1000000
        left = pd.DataFrame(causal_rich(original, "AUTHOR_1"))
        right = pd.DataFrame(causal_rich(modified, "AUTHOR_1"))
        pd.testing.assert_frame_equal(left.iloc[:6], right.iloc[:6])
        self.assertNotEqual(
            left.iloc[6]["past5_commentsCount"], right.iloc[6]["past5_commentsCount"]
        )

    def test_reposted_authors_are_excluded(self):
        posts = self.posts()
        repost = copy.deepcopy(posts[2])
        repost["author"]["code"] = "AUTHOR_2"
        posts.insert(3, repost)
        self.assertEqual(len(causal_rich(posts, "AUTHOR_1")), 8)

    def test_format_and_mentions_use_prior_posts(self):
        rows = causal_rich(self.posts(), "AUTHOR_1")
        self.assertEqual(rows[5]["graph_company_history_count"], 5)
        self.assertEqual(rows[5]["format0_history_count"], 5)
        self.assertEqual(rows[0]["graph_company_history_count"], 0)
        self.assertTrue(np.isnan(rows[0]["format0_history_delta"]))


if __name__ == "__main__":
    unittest.main()

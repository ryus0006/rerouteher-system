-- Soft-delete a test account: rename the login so it can no longer be signed into,
-- while keeping all data. Cascades rename saved_journey + interview_session too.
-- Restore by stripping the 'deleted-' prefix.
UPDATE rerouteher.app_user
SET username = 'deleted-' || username
WHERE username = 'REPLACE_WITH_TEST_USERNAME';

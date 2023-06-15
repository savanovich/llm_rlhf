Data collected from ["Information Security"](https://security.stackexchange.com/questions?tab=Votes) StackExchange subportal.

Query Editor Link: https://data.stackexchange.com/security/query/edit/1761719

``` sql
SELECT
top 10000
  answers.ParentId as [Post Link],
  questions.Id AS QuestionId,
  questions.Title AS QuestionTitle,
  questions.Body AS QuestionBody,
  questions.Score AS QuestionScore,
  questions.AnswerCount AS QuestionAnswersCount,
  answers.Id AS AnswerId,
  answers.Body AS AnswerBody,
  answers.Score AS AnswerScore
FROM
  Posts questions
LEFT JOIN
  Posts answers ON questions.Id = answers.ParentId
WHERE
  questions.AnswerCount > 2 AND
  questions.PostTypeId = 1
ORDER BY
  questions.Id, answers.Id
```




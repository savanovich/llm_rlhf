Project UI config Label Studio 

```xml
<View>
  <Header>Question #:</Header>
  <Text name="questionId" value="$question_id" />
  
  <TextArea name="question" value="$question" 
            toName="questionId"
            rows="10"
            editable="true" required="true" />
  
  <Header>Answer #:</Header>
  <Text name="answerId" value="$answer_id" />
  <TextArea name="answer" value="$answer" 
            toName="answerId"
            rows="10"
            editable="true" required="true" />
</View>```
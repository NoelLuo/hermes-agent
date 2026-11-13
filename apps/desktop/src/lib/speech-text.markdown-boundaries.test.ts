import { describe, expect, it } from 'vitest'

import { sanitizeTextForSpeech } from './speech-text'

// Same structure as the reported desktop read-aloud: bold, blank paragraphs,
// a lead-in and two hyphen bullets. No private conversation content.
const text = '**这是段落朗读**，不是发送延迟。\n\n现在检查：\n- 检查标题和列表。\n- 区分**音频静音**和等待下一段。\n\n保留正文和语速。'

describe('Markdown paragraph speech boundaries', () => {
  it('removes line structure before collapsing lines and respects CJK punctuation', () => {
    expect(sanitizeTextForSpeech(text)).toBe('这是段落朗读，不是发送延迟。 现在检查： 检查标题和列表。 区分音频静音和等待下一段。 保留正文和语速。')
  })

  it('keeps ordinary hyphens and strips task-list and heading markers', () => {
    expect(sanitizeTextForSpeech('## 提醒\n- [x] **完成**\n- [ ] 待办\n\nCOVID-19 stays.')).toBe('提醒 完成 待办. COVID-19 stays.')
  })

  it('does not speak control tokens', () => {
    expect(sanitizeTextForSpeech('你好。 [[qq_meme_optional:sample_id]] [[audio_as_voice]]')).toBe('你好。')
  })
})

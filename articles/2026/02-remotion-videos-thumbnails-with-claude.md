# Why I Use Remotion for Videos and Thumbnails (With Help From Claude)

*Published: February 2026 (Draft)*

---

I wanted to explore creating a video programmatically. It had animated text reveals, code snippets, feature cards sliding in, and transitions timed to a voiceover.

The usual path would be to fire up a video editor, drag assets around, and spend hours tweaking timings. I had been curious about Remotion for a while. This time I decided to dive in and see what it could really do.

---

## Discovering Remotion

Remotion is a framework that lets you create videos programmatically using React. You define scenes as components, control timing with frames, and render to MP4. Everything is code.

As a developer, I think in components and functions. Why should video creation be different?

There is a learning curve though. Remotion has its own patterns: `useCurrentFrame()`, `interpolate()`, `<Sequence>` components. You need to understand frame-based timing, spring animations, and how to structure compositions.

This is where Claude came in.

---

## How Claude Code Fits In

I have been using Claude Code extensively, and for this project, Claude Opus 4.5 became my collaborator.

The workflow is surprisingly natural. I describe what I want — "Create a scene where package names type out character by character with a blinking cursor, then badges fade in below." Claude generates the component, handles the frame calculations, and sets up the animations.

When something does not look right, I describe the issue. Claude iterates. We go back and forth until the timing feels natural.

What surprised me was how well it understood the visual intent behind my descriptions. I did not need to specify exact frame numbers or easing functions upfront. I would say "make it feel snappier" or "the text reveal should be more dramatic," and Claude would translate that into the right code.

But it goes beyond just writing what I ask for. I can say "I need this kind of video" and Claude will suggest animation approaches I had not considered. I pick the ones I like, and it starts building out the scenes. That back-and-forth is where the creative exploration happens.

---

## Matching Animations to Voiceover

This is where things got really interesting for me.

I started creating voiceovers separately and then matching the animations frame by frame to the narration. I tell Claude the duration of each segment — how long I spoke, where captions should appear, when transitions should hit — and it adjusts the frame counts accordingly.

Instead of manually calculating that a 3.5-second segment at 30fps needs exactly 105 frames, I just say "this scene should be 3.5 seconds to match the voiceover." Claude handles the math.

This is where Opus 4.5 really shines. It maintains context about which components make up each scene and what animations are already in place. When I ask it to adjust durations, it understands the full composition — which scene exists, what animation is already there — and knows exactly which frame values to update. That contextual awareness makes the collaboration feel seamless.

---

## Describing Animations in Natural Language

If you have seen an animation somewhere and want something similar, you can just describe it.

"I want the text to slide in from the left with a slight bounce, like that motion graphics style you see in tech product launches."

Claude tries to recreate it, gives you options, and you iterate from there. Getting to a working version is surprisingly fast. This is the kind of creative freedom that is hard to get in traditional video editors without deep motion design expertise. You are describing intent, not keyframes.

---

## The Unexpected Win: Thumbnails

Here is something I did not expect.

Remotion is not just for videos. You can render single frames as images. I realized I could use the same workflow for thumbnails.

Instead of opening Figma or Photoshop, I created a Thumbnail component. Same design system. Same fonts. Same color palette as the video. Perfect consistency.

```jsx
<Composition
  id="Thumbnail"
  component={Thumbnail}
  durationInFrames={1}
  fps={30}
  width={1280}
  height={720}
/>
```

One frame. Rendered as PNG. Done.

Multiple variants just by tweaking props. No manual export dance. No "save for web." Just `npx remotion still Thumbnail out/thumbnail.png`.

What makes this even better with Claude is the design collaboration. I describe what the thumbnail should contain and ask for layout suggestions, color combinations, ways to make it visually cohesive. Claude proposes layouts, I pick what works, and we refine. Some of the results genuinely surprised me.

---

## Why Not Canva?

I had previously used Canva for some of this — quick animations, thumbnail images, that kind of thing. Canva is easy, no question. It has AI features that help you be more creative, and for non-technical users it is a great choice.

But for someone on the technical side, Remotion with Claude Code has an edge.

In Canva, you are doing the layout work yourself within the tool's constraints. With Remotion, I can ask Claude to look at existing design ideas, suggest improvements, and generate the whole thing as a React component. I do not have to manually adjust anything in a GUI. I describe what I want and review the output.

And you do not have to understand the Remotion internals to use it. You just need to know the project is there. When you run it, Remotion opens a local web interface where you can preview every scene, play individual animations or the full composed video, see how long each section takes, and adjust accordingly.

That feedback loop of code, preview, iterate is incredibly fast.

---

## Try It Yourself

If you want to try this workflow, here is what I would suggest.

Start with Remotion's official templates — they cover everything from basic hello-world to TikTok-style captions and prompt-to-video generation. Run `npx create-video@latest` and you will have a working project in minutes.

I have also put together a small set of reusable starter components — the same ones I reference in this article:

- **CodeTyping** — Typing animation with blinking cursor. Configurable speed, colors, fonts.
- **TextReveal** — Word-by-word entrance with multiple styles: fade, slideUp, scaleIn.
- **FeatureCard** — Animated cards with staggered slide-in.
- **Thumbnail** — Single-frame composition for thumbnails.

Here is what the CodeTyping component looks like in practice:

```jsx
import { CodeTyping } from "./components/CodeTyping";

const CodeScene: React.FC = () => {
  return (
    <div style={{
      width: "100%", height: "100%",
      display: "flex", justifyContent: "center", alignItems: "center",
      background: "#0F0F1A", padding: 60,
    }}>
      <CodeTyping
        code={\`const greeting = "Hello, Remotion!";
console.log(greeting);\`}
        fontSize={22}
        typingSpeed={2}
      />
    </div>
  );
};
```

Copy the components into your project, wire them into your compositions, and start asking Claude to build scenes with them. Once Claude sees the patterns, it generates new variations that fit right in.

I went from zero Remotion experience to a polished video in a few days. Most of that time was spent on creative decisions, not technical struggles.

That is the kind of leverage I want from my tools.

---

*Built with Remotion and Claude Opus 4.5*
